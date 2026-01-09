def transpose(matrix):
    row=len(matrix)
    col=len(matrix[0])
    transpose=[]
    for j in range(col):
        new=[]
        for i in range(row):
            new.append(matrix[i][j])
        transpose.append(new)
    return transpose
row=int(input("Enter teh number of rows"))
col=int(input("Enter teh number if columns"))
matrix=[]
print("Enter the elements of matrix :")
for i in range(row):
    row=list(map(int,input("Enter the elements row wise").split(" ")))
    matrix.append(row)
result=transpose(matrix)
print("The transpose matrix:")
print(result)
