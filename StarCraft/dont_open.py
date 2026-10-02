with open("StarCraft/dont_open.txt", "w") as f:
    for i in range(1, 300001):
        f.write(str(i) + "\n")