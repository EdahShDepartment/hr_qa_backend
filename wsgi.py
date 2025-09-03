from waitress import serve
from app import app
import logging
import time

# 啟用 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('waitress')
logger.info("Starting server on http://127.0.0.1:5001")

# 自訂 WSGI middleware，用來紀錄請求時間與資訊
class RequestLoggerMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        start_time = time.time()

        def custom_start_response(status, headers, exc_info=None):
            # 請求結束後，記錄請求資訊
            duration = time.time() - start_time
            method = environ.get('REQUEST_METHOD')
            path = environ.get('PATH_INFO')
            client_ip = environ.get('REMOTE_ADDR')
            logger.info(f'{client_ip} - "{method} {path}" {status} - {duration:.4f}s')
            return start_response(status, headers, exc_info)

        return self.app(environ, custom_start_response)

# 包裝 middleware
logged_app = RequestLoggerMiddleware(app)

if __name__ == '__main__':
    serve(
        logged_app,
        host='127.0.0.1',
        port=5001,
        threads=4,
        connection_limit=200,
        backlog=100
    )