

class SandboxError(RuntimeError):
    pass


class SandboxPolicyError(SandboxError):
    pass


class SandboxUnavailableError(SandboxError):
    pass


__all__ = ["SandboxError", "SandboxPolicyError", "SandboxUnavailableError"]
