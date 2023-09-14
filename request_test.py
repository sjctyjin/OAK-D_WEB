import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
#
# # 忽略SSL證書驗證警告
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


# url = 'https://localhost:44301/WebForm1.aspx/Web_SetCoords'
url = 'http://192.168.2.105:8081/WebForm1.aspx/Web_SetCoords'

# 準備要發送的數據，這裡假設您要傳遞一個JSON對象
data = {'cordX': 762.922,'cordY':-298.123,'cordZ':763.068}

# 使用requests庫發送POST請求
response = requests.post(url, json=data, verify=False)

# 檢查請求是否成功
if response.status_code == 200:
    print('POST請求成功')
    print('伺服器回應：', response.text)
else:
    print('POST請求失敗')
