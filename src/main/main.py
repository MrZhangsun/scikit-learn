# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    color = pd.DataFrame({
        "color": ["红", "绿", "蓝"]
    })
    dummy_encoded = pd.get_dummies(color)
    print(dummy_encoded)

def one_hot_encoding():
    df = pd.DataFrame({
        "面积": [70, 80, 90],
        "城市": ["上海", "北京", "广州"],
        "价格": [100, 120, 130]
    })

    dummy_encoded = pd.get_dummies(df)
    print(dummy_encoded)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    one_hot_encoding()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
