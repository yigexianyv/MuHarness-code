

class MCPError(RuntimeError):
    pass


class MCPConfigurationError(MCPError):
    pass


class MCPConnectionError(MCPError):
    pass


class MCPToolDiscoveryError(MCPError):
    pass


class MCPToolCallError(MCPError):
    pass


__all__ = [
    "MCPConfigurationError",
    "MCPConnectionError",
    "MCPError",
    "MCPToolCallError",
    "MCPToolDiscoveryError",
]
