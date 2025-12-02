import re
with open('MIST_test.ijm') as f:
    t = f.read().strip('run(').strip(');')
    l = t.split(', ')[1].strip('"').split()
    print('\n'.join(l),len(l))
    t = re.sub('imagedir=.* filenamepattern=','imagedir=somethingelse filenamepattern=',t)
    t = re.sub('outputpath=.* display','outputpath=somethingelse display',t)
    t = re.sub('gridwidth=.* gridheight=.* starttile','gridwidth=5 gridheight=5 starttile',t)
    t = re.sub('extentwidth=.* extentheight=.* timeslices=','extentwidth=5 extentheight=5 timeslices=',t)
