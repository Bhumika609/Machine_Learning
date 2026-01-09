def multiply(a,b):
    row_a=len(a)
    col_a=len(a[0])
    row_b=len(b)
    col_b=len(b[0])
    if row_a!=col_b:
        print("Error")
        return
    result=[[0 for _ in range(col_b)]for _ in range(row_a)]
    for i in range(row_a):
        for j in range(col_b):
            for k in range(col_a):
                result[i][j]=result[i][j]+a[i][k]*b[k][j]
    return result
a=[[1,2,3],[4,5,6]]
b=[[7,8],[9,10],[10,11]]
product=multiply(a,b)
print("Product of the matrices is",product)