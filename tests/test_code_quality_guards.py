import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_kernel_method_call(node: ast.AST, method_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "kernel"
        and node.func.attr == method_name
    )


def test_a_memorix_has_no_silent_broad_exception_handlers() -> None:
    violations = []
    source_paths = list((REPO_ROOT / "src" / "a_memorix").rglob("*.py"))
    assert source_paths

    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if len(node.body) != 1 or not isinstance(node.body[0], ast.Pass):
                continue
            if node.type is None or (
                isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
            ):
                violations.append(f"{path}:{node.lineno}")

    assert violations == []
