def common(list_1,list_2):
    count_number=0
    for i in range(len(list_1)):
        for j in range(len(list_2)):
            if list_1[i]==list_2[j]:
                count_number=count_number+1
    return count_number
list_1=list(map(int,input("Enter the elements of the first list").split(" ")))
list_2=list(map(int,input("Enter the lemets of the secind list").split(" ")))
common_elements_count=common(list_1,list_2)
print("The number of the common elements in both the lists",common_elements_count)
