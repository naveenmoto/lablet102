import os, rarfile
def unrar():
	with open("./rarlist", "r") as f:
		rar_files = f.readlines()
	for fname in rar_files:
		os.system("wget " + fname)
	os.system("mv *.rar week2/")
	for rf in os.listdir("./week2"):
		f = os.path.join("./week2",rf)
		o = rarfile.RarFile(f)
		for u in o.infolist():
			print (u.filename, u.file_size)
			o.extractall("./week2")
	
if __name__ == "__main__":
	unrar()