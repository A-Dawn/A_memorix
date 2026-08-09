package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	api "github.com/MaiM-with-u/A_memorix/gen/go/a_memorix/api/v1"
	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
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
	listenAddress := flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
	grpcTarget := flag.String("grpc-target", "127.0.0.1:50051", "A_memorix gRPC target")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	connection, err := grpc.NewClient(
		*grpcTarget,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		log.Fatalf("create gRPC client: %v", err)
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
		log.Fatalf("register namespace gateway: %v", err)
	}
	if err := api.RegisterBackupServiceHandler(ctx, mux, connection); err != nil {
		log.Fatalf("register backup gateway: %v", err)
	}
	if err := api.RegisterAuthServiceHandler(ctx, mux, connection); err != nil {
		log.Fatalf("register auth gateway: %v", err)
	}
	if err := api.RegisterMemoryServiceHandler(ctx, mux, connection); err != nil {
		log.Fatalf("register memory gateway: %v", err)
	}
	if err := api.RegisterJobServiceHandler(ctx, mux, connection); err != nil {
		log.Fatalf("register job gateway: %v", err)
	}

	server := &http.Server{
		Addr:              *listenAddress,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	serverError := make(chan error, 1)
	go func() {
		log.Printf("A_memorix HTTP/JSON gateway listening on %s", *listenAddress)
		serverError <- server.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownContext); err != nil {
			log.Printf("gateway shutdown failed: %v", err)
		}
	case err := <-serverError:
		if !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("gateway server failed: %v", err)
		}
	}
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
		log.Printf("encode HTTP error: %v", err)
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
