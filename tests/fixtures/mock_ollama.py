"""Synthetic local protocol fixture; never use this as a real model."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/api/generate':
            self.send_error(404)
            return
        size=int(self.headers.get('Content-Length', '0'))
        request=json.loads(self.rfile.read(size))
        result={'status':'yes','rationale':'合成示例：政策证据。','evidence':'政策证据'}
        data=json.dumps({'model':request['model'],'done':True,'done_reason':'stop','response':json.dumps(result,ensure_ascii=False)},ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *_args):
        pass

if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1',18513),Handler).serve_forever()
