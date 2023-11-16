if __name__ == "__main__":
    with open('curve-left_processed.txt', "r") as file:
        lines = file.readlines()[4:-2]

    output = []
    for line in lines:
        # print(line)
        array_str = line.split()
        array = [float(num.strip()) for num in line[1:-2].split(",")]

        modified_array = [num * 1000 for num in array[:3]]

        # Append the modified array to the output
        output.append(modified_array)


    with open('curve-1000-left.txt', 'w') as file2:
        for modified_array in output:
            # file2.write(modified_array)
            # Convert modified array elements to strings before writing
            modified_array_str = [str(num) for num in modified_array]
            file2.write('[')
            file2.write(', '.join(modified_array_str) + ']\n')
            # file2.write(']')