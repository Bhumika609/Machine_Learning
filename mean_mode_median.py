import random
def generate_numbers(n,start,end):
    numbers=[]
    for _ in range(n):
        numbers.append(random.randint(start,end))
    return numbers
def mean(numbers):
    num_sum=sum(numbers)
    return num_sum/len(numbers)
def median(numbers):  
    sort=sorted(numbers)
    n=len(sort)
    if n%2==0:
        median=(sort[n//2-1]+sort[n//2])/2
    else:
        median=sort[n//2]
    return median
def mode(numbers):
    freq={}
    for num in numbers:
        if num in freq:
            freq[num]=freq[num]+1
        else:
            freq[num]=1
    max_count=max(freq.values())
    modes=[]
    for num in freq:
        if freq[num]==max_count:
            modes.append(num)
    return modes
numbers=generate_numbers(100,100,150)
mean=mean(numbers)
median=median(numbers)
mode=mode(numbers)
print("The numbers are",list(numbers))
print("The mean of the numbers",mean)
print("The median of the numbers",median)
print("The mode of the numbers",mode)
