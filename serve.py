#!flask/bin/python
import argparse

from app import create_app

def parse_args():
    # 创建一个parser对象
    parser = argparse.ArgumentParser(description='parser demo')
 
    parser.add_argument('--port', '-p', type=int, default=8000, help = 'port to listen on')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args = parser.parse_args()
 
    return args
 

opt = parse_args()
app = create_app()
app.run(debug=opt.debug, host="0.0.0.0", port=opt.port)
