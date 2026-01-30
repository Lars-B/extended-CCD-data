import csv

csv_file = "subset-OAW.csv"

names = set()

with open(csv_file, newline="") as f:
	reader = csv.DictReader(f)
	for row in reader:
		names.add(row["TAXON"])

print(f"Read {len(names)} names from csv")

# fasta_in = "../08022022_WGS_ALIGNMENT.fa"
# fasta_out = "filtered.fa"

#with open(fasta_in, "r") as fin, open(fasta_out, "w") as fout:
#    write_seq = False
#
#    for line in fin:
#        if line.startswith(">"):
#            header = line[1:].strip()
#            seq_name = header.split()[0]
#
#            write_seq = seq_name in names
#
#            if write_seq:
#                fout.write(line)
#        else:
#            if write_seq:
#                fout.write(line)

import xml.etree.ElementTree as ET

input_xml = "../08072022_CDS_NCR_STRICT_LINKED.txt"
output_csv = "dates.txt"

tree = ET.parse(input_xml)
root = tree.getroot()

value_str = root[-1][0][0][0].attrib["value"]

rows = []
for item in value_str.split(","):
    name, date = item.split("=", 1)
    rows.append((name, date))

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["name", "date"])
    writer.writerows(rows)
