# predictor/middleware.py
"""
Custom middleware for caching and rate limiting
"""

from django.core.cache import cache
from django.http import HttpResponse, JsonResponse
import hashlib
import json


class PredictionCacheMiddleware:
    """
    Cache prediction results to reduce ML inference load
    Uses Redis to store predictions for identical inputs
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Only cache prediction endpoint
        if request.path == '/predict/' and request.method == 'POST':
            # Generate cache key from request data
            cache_key = self._generate_cache_key(request)

            # Try to get cached prediction
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                # Add cache hit header
                response = self.get_response(request)
                response['X-Cache'] = 'HIT'
                return response

            # Process request normally
            response = self.get_response(request)

            # Cache the response (1 hour TTL)
            if response.status_code == 200:
                cache.set(cache_key, True, 3600)
                response['X-Cache'] = 'MISS'

            return response

        return self.get_response(request)

    def _generate_cache_key(self, request):
        """Generate MD5 hash of POST data for cache key"""
        try:
            # Create deterministic representation of POST data
            post_data = dict(request.POST.items())
            # Remove CSRF token from cache key
            post_data.pop('csrfmiddlewaretoken', None)

            # Sort keys for consistency
            sorted_data = json.dumps(post_data, sort_keys=True)

            # Generate hash
            cache_key = f"prediction:{hashlib.md5(sorted_data.encode()).hexdigest()}"
            return cache_key
        except:
            # If hashing fails, don't cache
            return None


class RateLimitMiddleware:
    """
    Simple rate limiting based on IP address
    Prevents abuse and DoS attacks
    """
    def __init__(self, get_response):
        self.get_response = get_response
        self.max_requests = 100  # requests per hour
        self.window = 3600  # 1 hour in seconds

    def __call__(self, request):
        # Get client IP
        ip = self._get_client_ip(request)

        # Rate limit key
        rate_key = f"rate_limit:{ip}"

        # Get current request count
        request_count = cache.get(rate_key, 0)

        if request_count >= self.max_requests:
            return JsonResponse({
                'error': 'Rate limit exceeded',
                'message': f'Maximum {self.max_requests} requests per hour allowed',
                'retry_after': cache.ttl(rate_key)
            }, status=429)

        # Increment counter
        if request_count == 0:
            # First request in window
            cache.set(rate_key, 1, self.window)
        else:
            cache.incr(rate_key)

        # Process request
        response = self.get_response(request)

        # Add rate limit headers
        response['X-RateLimit-Limit'] = str(self.max_requests)
        response['X-RateLimit-Remaining'] = str(self.max_requests - request_count - 1)

        return response

    def _get_client_ip(self, request):
        """Extract client IP from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
