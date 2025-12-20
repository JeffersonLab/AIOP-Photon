import csv

def write_csv(filename, rows, header):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
