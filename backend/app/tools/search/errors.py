

class SearchError(RuntimeError):
    pass


class SearchAuthenticationError(SearchError):
    pass


class SearchRateLimitError(SearchError):
    pass


class SearchNetworkError(SearchError):
    pass


class SearchNoResultsError(SearchError):
    pass


class SearchUnavailableError(SearchError):
    pass


__all__ = [
    "SearchAuthenticationError",
    "SearchError",
    "SearchNetworkError",
    "SearchNoResultsError",
    "SearchRateLimitError",
    "SearchUnavailableError",
]
