import glob

lines = []
files = glob.glob("corpus/*.txt")
for fle in files:
    with open(fle) as f:
        lines += f.readlines()
print(lines)


# lines = []
# #test = []
# files = glob.glob("/corpus/*.txt")
# for fle in files:
#     with open(fle) as f:
#         lines += f.readlines()        
#     	#test = np.array('lines')
# #print(test)
# print(lines)