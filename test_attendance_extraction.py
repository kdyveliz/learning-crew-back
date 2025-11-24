import pandas as pd
import os
from file_utils import extract_attendance_list_from_excel


def test_extract_attendance_list():
    # Create a dummy Excel file
    data = {
        "순번": [1, 2, 3],
        "참석명단": ["홍길동", "김철수", "이영희"],
        "비고": ["", "", ""],
    }
    df = pd.DataFrame(data)
    file_path = "test_attendance.xlsx"
    df.to_excel(file_path, index=False)

    try:
        # Test extraction
        attendance_list = extract_attendance_list_from_excel(file_path)
        print(f"Extracted list: {attendance_list}")

        expected_list = ["홍길동", "김철수", "이영희"]
        assert (
            attendance_list == expected_list
        ), f"Expected {expected_list}, but got {attendance_list}"
        print("Test passed!")

    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    test_extract_attendance_list()
