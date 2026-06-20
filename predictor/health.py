# predictor/health.py
"""
Health check endpoint for monitoring and load balancers
"""

from django.http import JsonResponse
from django.db import connection
from django.core.cache import cache
from datetime import datetime
import sys


def health_check(request):
    """
    Comprehensive health check endpoint
    Returns 200 if all systems operational, 503 otherwise

    Used by:
    - Docker healthcheck
    - Load balancers (ALB, ELB)
    - Monitoring systems (Datadog, New Relic)
    - Kubernetes readiness/liveness probes
    """
    health = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'tennis-predictor',
        'version': '1.0.0',
        'checks': {}
    }

    # 1. Database connectivity check
    try:
        connection.ensure_connection()
        health['checks']['database'] = {
            'status': 'ok',
            'message': 'Database connection successful'
        }
    except Exception as e:
        health['checks']['database'] = {
            'status': 'error',
            'message': str(e)
        }
        health['status'] = 'unhealthy'

    # 2. Redis cache check
    try:
        cache.set('health_check_test', 'ok', 10)
        test_value = cache.get('health_check_test')
        if test_value == 'ok':
            health['checks']['redis'] = {
                'status': 'ok',
                'message': 'Cache operational'
            }
        else:
            raise Exception("Cache read/write mismatch")
    except Exception as e:
        health['checks']['redis'] = {
            'status': 'error',
            'message': str(e)
        }
        health['status'] = 'degraded'  # Not critical, can continue without cache

    # 3. Model availability check
    try:
        from .predictor_utils import MODELS

        models_status = {
            'enhanced_atp': MODELS.get('enhanced_atp') is not None,
            'enhanced_wta': MODELS.get('enhanced_wta') is not None,
            'basic_atp': MODELS.get('basic_atp') is not None,
            'basic_wta': MODELS.get('basic_wta') is not None,
        }

        # At least one model type should be available
        has_models = any(models_status.values())

        health['checks']['models'] = {
            'status': 'ok' if has_models else 'warning',
            'message': 'Models loaded successfully' if has_models else 'No models available',
            'details': models_status
        }

        if not has_models:
            health['status'] = 'degraded'

    except Exception as e:
        health['checks']['models'] = {
            'status': 'error',
            'message': str(e)
        }
        health['status'] = 'degraded'

    # 4. Python version info
    health['python_version'] = sys.version.split()[0]

    # 5. System resources (basic check)
    try:
        import os
        health['process_id'] = os.getpid()
    except:
        pass

    # Determine HTTP status code
    if health['status'] == 'healthy':
        status_code = 200
    elif health['status'] == 'degraded':
        status_code = 200  # Still operational, but with warnings
    else:
        status_code = 503  # Service unavailable

    return JsonResponse(health, status=status_code)


def readiness_check(request):
    """
    Kubernetes readiness probe
    Returns 200 when service is ready to accept traffic
    """
    try:
        # Quick database check
        connection.ensure_connection()
        return JsonResponse({'status': 'ready'}, status=200)
    except:
        return JsonResponse({'status': 'not ready'}, status=503)


def liveness_check(request):
    """
    Kubernetes liveness probe
    Returns 200 if service is alive (even if not fully functional)
    """
    return JsonResponse({'status': 'alive'}, status=200)
