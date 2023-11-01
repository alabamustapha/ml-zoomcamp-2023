from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return "PONG"

if __name__ == '__main__':
    app.run(debug=True, port=9696, host='0.0.0.0')