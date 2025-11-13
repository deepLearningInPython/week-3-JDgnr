import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
    pass
    input_length = len(input_array)
    kernel_length = len(kernel_array)
    return input_length - kernel_length + 1

# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    # Length of the output using the formula
    output_length = len(input_array) - len(kernel_array) + 1
    
    # Initialize output array
    output = np.zeros(output_length)
    
    # Flip kernel for true convolution (as opposed to cross-correlation)
    kernel_flipped = kernel_array[::-1]
    
    # Sliding window
    for i in range(output_length):
        window = input_array[i : i + len(kernel_array)]
        output[i] = np.sum(window * kernel_flipped)
    
    return output

# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    pass input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel_matrix.shape
    
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    
    return (output_height, output_width)


# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    # Tip: same tips as above, but you might need a nested loop here in order to
    # define which parts of the input matrix need to be multiplied with the kernel matrix.
    pass # Get output dimensions
    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel_matrix.shape
    
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1
    
    # Prepare output matrix
    output = np.zeros((output_h, output_w))
    
    # Flip kernel for true convolution
    kernel_flipped = np.flipud(np.fliplr(kernel_matrix))
    
    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            # Extract window of the input
            window = input_matrix[i : i + kernel_h, j : j + kernel_w]
            # Elementwise multiply and sum
            output[i, j] = np.sum(window * kernel_flipped)

    return output


# -----------------------------------------------