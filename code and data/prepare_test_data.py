import csv
with open('test_data_english_spam.txt', 'r', encoding='ansi') as f:
    content = f.readlines()
content = [x.strip() for x in content]

with open('test_sms.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    for line in content:
        writer.writerow([line])
