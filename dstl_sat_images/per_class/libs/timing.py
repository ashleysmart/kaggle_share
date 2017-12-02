import datetime

def sync():
    return datetime.datetime.now()

def timing(mark):
    new_mark = datetime.datetime.now()
    delta = new_mark - mark
    print "                                                        .... ", delta
    return mark
