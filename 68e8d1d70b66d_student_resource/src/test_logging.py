import sys

try:
    with open("C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/src/test.log", "w") as f:
        f.write("Test log entry")
except Exception as e:
    with open("C:/Users/HEMA HARSHINI/Desktop/ml/MLChallenge/68e8d1d70b66d_student_resource/src/test_error.log", "w") as f:
        f.write(str(e))