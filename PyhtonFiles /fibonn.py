def square_numbers(nums):
    for i in nums:
        yield i
        b = yield 
        
        print(b)
        
'''
# passing this numbers, for this number
#yield those numbers and do the following
'''
for j in range(100):

    my_nums = square_numbers([j]) 

    for num in my_nums:
        print num
