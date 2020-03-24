import Tkinter as tk
import requests
import json
import time


class User(object):
    def __init__(self, follow_num, uri, name=""):
        self.name = name
        self.follow_num = follow_num
        self.url = uri


def query():
    var = e.get("0.0", "end")
    host = 'http://139.196.22.4:30003/v1/verify/testcp'
    user_info = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=%s&containerid=100505%s'
    response = requests.post(host, data=var.encode("utf-8"))
    json_dict = json.loads(response.text)
    urls = []
    m = {}
    for k, v in json_dict.items():
        if k == "data":
            for url in v:
                data = url.get("weiboUrl", -1)
                num = url.get("followNum", -1)
                if data == -1 or num == -1:
                    break
                urls.append(data[data.index('u/') + 2:])
                m[data[data.index('u/') + 2:]] = User(num, data)
    t.delete("0.0", "end")
    t.insert('end', 'total count:%d\n' % (len(urls)))
    count = 1
    for u in urls:
        fill_host = user_info % (u, u)
        res = requests.get(fill_host)
        time.sleep(0.1)
        json_user_name = json.loads(res.text)
        ob = m.get(u, -1)
        if ob != -1:
            t.insert('end', "%-3d: %-40s fans: %-10d \n" % (
                count, json_user_name.get("userInfo").get("screen_name"), ob.follow_num))
            t.update()
            count = count + 1


if __name__ == '__main__':
    window = tk.Tk()
    window.title('my window')
    window.geometry('600x600')
    e = tk.Text(window, width=600, height=20)
    e.pack()

    b1 = tk.Button(window, text='query', width=15, height=2, command=query)
    b1.pack()
    t = tk.Text(window, height=2, width=100)
    s1 = tk.Scrollbar(t)
    s1.config(command=t.yview)
    t.config(yscrollcommand=s1.set, width=20, height=20, background='#ffffff')
    s1.pack(side='right', fill='y')
    t.pack(side='left', fill='both', expand=1)
    window.mainloop()

## people change
