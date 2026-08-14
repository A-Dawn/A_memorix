package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	api "github.com/A-Dawn/A_memorix/gen/go/a_memorix/api/v1"
	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	grpcHealth "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/structpb"
)

type errorEnvelope struct {
	Code      string         `json:"code"`
	Message   string         `json:"message"`
	RequestID string         `json:"requestId,omitempty"`
	TraceID   string         `json:"traceId,omitempty"`
	Retryable bool           `json:"retryable"`
	Details   map[string]any `json:"details"`
}

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))
	listenAddress := flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
	grpcTarget := flag.String("grpc-target", "127.0.0.1:50051", "A_memorix gRPC target")
	grpcCA := flag.String("grpc-ca", "", "CA certificate for the gRPC backend")
	grpcServerName := flag.String("grpc-server-name", "", "TLS server name for the gRPC backend")
	grpcClientCert := flag.String("grpc-client-cert", "", "client certificate for backend mTLS")
	grpcClientKey := flag.String("grpc-client-key", "", "client private key for backend mTLS")
	tlsCert := flag.String("tls-cert", "", "HTTPS server certificate")
	tlsKey := flag.String("tls-key", "", "HTTPS server private key")
	tlsClientCA := flag.String("tls-client-ca", "", "CA certificate required from HTTPS clients")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	transportCredentials, err := backendCredentials(
		*grpcCA,
		*grpcServerName,
		*grpcClientCert,
		*grpcClientKey,
	)
	if err != nil {
		slog.Error("configure gRPC backend TLS", "error", err)
		os.Exit(1)
	}
	connection, err := grpc.NewClient(*grpcTarget, grpc.WithTransportCredentials(transportCredentials))
	if err != nil {
		slog.Error("create gRPC client", "error", err)
		os.Exit(1)
	}
	defer connection.Close()

	mux := runtime.NewServeMux(
		runtime.WithMarshalerOption(runtime.MIMEWildcard, &runtime.JSONPb{
			MarshalOptions: protojson.MarshalOptions{},
			UnmarshalOptions: protojson.UnmarshalOptions{
				DiscardUnknown: false,
			},
		}),
		runtime.WithIncomingHeaderMatcher(headerMatcher),
		runtime.WithErrorHandler(handleHTTPError),
		runtime.WithDisableHTTPMethodOverride(),
		runtime.WithDisablePathLengthFallback(),
	)
	if err := api.RegisterNamespaceServiceHandler(ctx, mux, connection); err != nil {
		slog.Error("register namespace gateway", "error", err)
		os.Exit(1)
	}
	if err := api.RegisterBackupServiceHandler(ctx, mux, connection); err != nil {
		slog.Error("register backup gateway", "error", err)
		os.Exit(1)
	}
	if err := api.RegisterAuthServiceHandler(ctx, mux, connection); err != nil {
		slog.Error("register auth gateway", "error", err)
		os.Exit(1)
	}
	if err := api.RegisterMemoryServiceHandler(ctx, mux, connection); err != nil {
		slog.Error("register memory gateway", "error", err)
		os.Exit(1)
	}
	if err := api.RegisterJobServiceHandler(ctx, mux, connection); err != nil {
		slog.Error("register job gateway", "error", err)
		os.Exit(1)
	}

	rootMux := http.NewServeMux()
	rootMux.HandleFunc("GET /healthz", healthHandler(grpcHealth.NewHealthClient(connection)))
	rootMux.Handle("/", mux)
	tlsConfig, err := serverTLSConfig(*tlsCert, *tlsKey, *tlsClientCA)
	if err != nil {
		slog.Error("configure HTTPS", "error", err)
		os.Exit(1)
	}

	server := &http.Server{
		Addr:              *listenAddress,
		Handler:           accessLog(rootMux),
		TLSConfig:         tlsConfig,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	serverError := make(chan error, 1)
	go func() {
		slog.Info("A_memorix HTTP/JSON gateway listening", "address", *listenAddress, "tls", *tlsCert != "")
		if *tlsCert != "" {
			serverError <- server.ListenAndServeTLS(*tlsCert, *tlsKey)
			return
		}
		serverError <- server.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownContext); err != nil {
			slog.Error("gateway shutdown failed", "error", err)
		}
	case err := <-serverError:
		if !errors.Is(err, http.ErrServerClosed) {
			slog.Error("gateway server failed", "error", err)
			os.Exit(1)
		}
	}
}

type responseRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (recorder *responseRecorder) WriteHeader(status int) {
	if recorder.wroteHeader {
		return
	}
	recorder.status = status
	recorder.wroteHeader = true
	recorder.ResponseWriter.WriteHeader(status)
}

func (recorder *responseRecorder) Unwrap() http.ResponseWriter {
	return recorder.ResponseWriter
}

func (recorder *responseRecorder) Flush() {
	if !recorder.wroteHeader {
		recorder.WriteHeader(http.StatusOK)
	}
	_ = http.NewResponseController(recorder.ResponseWriter).Flush()
}

func accessLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		started := time.Now()
		recorder := &responseRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(recorder, request)
		slog.Info(
			"HTTP request completed",
			"method", request.Method,
			"path", request.URL.Path,
			"status", recorder.status,
			"duration_ms", float64(time.Since(started).Microseconds())/1000,
			"request_id", request.Header.Get("X-Request-ID"),
		)
	})
}

func headerMatcher(key string) (string, bool) {
	switch strings.ToLower(key) {
	case "idempotency-key", "x-request-id", "x-trace-id":
		return strings.ToLower(key), true
	default:
		return runtime.DefaultHeaderMatcher(key)
	}
}

func handleHTTPError(
	_ context.Context,
	_ *runtime.ServeMux,
	_ runtime.Marshaler,
	w http.ResponseWriter,
	_ *http.Request,
	err error,
) {
	grpcStatus := status.Convert(err)
	envelope := errorEnvelope{
		Code:      stableErrorCode(grpcStatus.Code()),
		Message:   grpcStatus.Message(),
		Retryable: grpcStatus.Code() == codes.Unavailable,
		Details:   map[string]any{},
	}
	for _, value := range grpcStatus.Details() {
		detail, ok := value.(*api.ErrorDetail)
		if !ok {
			continue
		}
		envelope.Code = detail.GetCode()
		envelope.RequestID = detail.GetRequestId()
		envelope.TraceID = detail.GetTraceId()
		envelope.Retryable = detail.GetRetryable()
		envelope.Details = structMap(detail.GetDetails())
		break
	}

	w.Header().Set("Content-Type", "application/json")
	if envelope.RequestID != "" {
		w.Header().Set("X-Request-ID", envelope.RequestID)
	}
	w.WriteHeader(runtime.HTTPStatusFromCode(grpcStatus.Code()))
	if err := json.NewEncoder(w).Encode(envelope); err != nil {
		slog.Error("encode HTTP error", "error", err)
	}
}

func backendCredentials(
	caFile string,
	serverName string,
	clientCertFile string,
	clientKeyFile string,
) (credentials.TransportCredentials, error) {
	if caFile == "" && serverName == "" && clientCertFile == "" && clientKeyFile == "" {
		return insecure.NewCredentials(), nil
	}
	if (clientCertFile == "") != (clientKeyFile == "") {
		return nil, errors.New("gRPC client certificate and private key must be set together")
	}
	roots, err := x509.SystemCertPool()
	if err != nil || roots == nil {
		roots = x509.NewCertPool()
	}
	if caFile != "" {
		payload, readErr := os.ReadFile(caFile)
		if readErr != nil {
			return nil, fmt.Errorf("read gRPC CA: %w", readErr)
		}
		if !roots.AppendCertsFromPEM(payload) {
			return nil, errors.New("gRPC CA does not contain a certificate")
		}
	}
	config := &tls.Config{
		MinVersion: tls.VersionTLS12,
		RootCAs:    roots,
		ServerName: serverName,
	}
	if clientCertFile != "" {
		certificate, loadErr := tls.LoadX509KeyPair(clientCertFile, clientKeyFile)
		if loadErr != nil {
			return nil, fmt.Errorf("load gRPC client certificate: %w", loadErr)
		}
		config.Certificates = []tls.Certificate{certificate}
	}
	return credentials.NewTLS(config), nil
}

func serverTLSConfig(certFile string, keyFile string, clientCAFile string) (*tls.Config, error) {
	if (certFile == "") != (keyFile == "") {
		return nil, errors.New("HTTPS certificate and private key must be set together")
	}
	if certFile == "" {
		if clientCAFile != "" {
			return nil, errors.New("HTTPS client CA requires HTTPS to be enabled")
		}
		return nil, nil
	}
	config := &tls.Config{MinVersion: tls.VersionTLS12}
	if clientCAFile == "" {
		return config, nil
	}
	payload, err := os.ReadFile(clientCAFile)
	if err != nil {
		return nil, fmt.Errorf("read HTTPS client CA: %w", err)
	}
	clientRoots := x509.NewCertPool()
	if !clientRoots.AppendCertsFromPEM(payload) {
		return nil, errors.New("HTTPS client CA does not contain a certificate")
	}
	config.ClientCAs = clientRoots
	config.ClientAuth = tls.RequireAndVerifyClientCert
	return config, nil
}

func healthHandler(client grpcHealth.HealthClient) http.HandlerFunc {
	return func(w http.ResponseWriter, request *http.Request) {
		ctx, cancel := context.WithTimeout(request.Context(), 2*time.Second)
		defer cancel()
		response, err := client.Check(ctx, &grpcHealth.HealthCheckRequest{})
		if err != nil || response.GetStatus() != grpcHealth.HealthCheckResponse_SERVING {
			http.Error(w, "not serving", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("serving\n"))
	}
}

func structMap(value *structpb.Struct) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	return value.AsMap()
}

func stableErrorCode(code codes.Code) string {
	switch code {
	case codes.InvalidArgument:
		return "invalid_argument"
	case codes.Unauthenticated:
		return "unauthorized"
	case codes.PermissionDenied:
		return "forbidden"
	case codes.NotFound:
		return "not_found"
	case codes.AlreadyExists, codes.Aborted:
		return "conflict"
	case codes.DataLoss:
		return "integrity_error"
	case codes.FailedPrecondition:
		return "migration_required"
	case codes.Unavailable, codes.Unimplemented:
		return "capability_unavailable"
	case codes.DeadlineExceeded:
		return "timeout"
	case codes.Canceled:
		return "cancelled"
	default:
		return "internal_error"
	}
}
