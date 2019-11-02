for i in range(10):
    name = i
    if(name<10):
        name = str(name)
        name = name.zfill(4)
    else:
        name = str(name)
    print(name)
