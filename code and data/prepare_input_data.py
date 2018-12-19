import csv
with open('train_english_spam.txt', 'r') as f:
    content = f.readlines()

content = [x.strip() for x in content]
c1 = []
c2 = []
for line in content:
    t = line.rsplit(',', 1)
    c1.append(t[0])
    c2.append(t[1])

with open('labelled_input.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(zip(c1, c2))
