import requests
import json


def send_api(direction, method):
    url = "https://oum9lh2tzc.execute-api.ap-northeast-2.amazonaws.com/cart/cart"

    headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}
    body = {
        "state": {
            "direction": direction
        }
    }

    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers,
                                     data=json.dumps(body, ensure_ascii=False, indent="\t"))
        print("response status %r" % response.status_code)
        print("response text %r" % response.text)
    except Exception as ex:
        print(ex)  # 호출 예시 send_api("/test", "POST")


#send_api("center","POST")
