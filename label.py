class Label:

    def __init__(self, path):
        f = open(path, "r")
        line = f.readline()[:-1] # Rm last character - \n
        self.index = dict()
        while(line):
            label, start, end = line.split(',')
            if label not in self.index:
                self.index[label] = [range(int(start), int(end)+1)]
            else:  
                self.index[label].append(range(int(start), int(end)+1))
            line = f.readline()[:-1] # Rm last char

    # Given a time, find the label that it goes with
    def getLabel(self, time):
        for k in self.index.keys():
            for r in self.index[k]:
                if time in r:
                    if k == 'Stationary':
                        lab=0
                    elif k == 'Walking-flat-surface':
                        lab=1
                    elif k == 'Walking-up-stairs':
                        lab=2
                    elif k == 'Walking-down-stairs':
                        lab=3
                    elif k == 'Elevator-up':
                        lab=4
                    elif k == 'Elevator-down':
                        lab=5
                    elif k == 'Running':
                        lab=6
                    return lab
