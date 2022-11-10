import pandas as pd
import txt_util as txt_util
import re
import os


dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

res_list = list()


def make_list(cat, file_name):
    list_res = list()
    text_lines = txt_util.TxtHolder(file_path=dir + file_name).read_txt_file(as_lines=True)
    for line in text_lines:
        text = re.sub(r'^\d+.\s', '', line).strip()
        if len(text) == 0:
            continue
        tmp_dict = dict()
        tmp_dict["text"] = text
        tmp_dict["cat"] = cat
        list_res.append(tmp_dict)
    return list_res


res_list.extend(make_list(0, r"List other.txt"))
res_list.extend(make_list(1, r"List apology.txt"))
res_list.extend(make_list(2, r"List request.txt"))
res_list.extend(make_list(3, r"List greetings.txt"))
res_list.extend(make_list(4, r"list complaints.txt"))
res_list.extend(make_list(5, r"list compliments.txt"))

df = pd.DataFrame(res_list)
path = dir + r"Exp_25_full_list.csv"
df.to_csv(path, index=True, index_label="index", encoding="utf_8_sig")
