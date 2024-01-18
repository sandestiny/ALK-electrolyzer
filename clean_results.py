# # # # # # # # #
# Clean Results #
# # # # # # # # #
#
import os

dir_path = os.path.dirname(__file__)
file_list = os.listdir(dir_path)

if 'clean_results.py' in file_list:
    for item in file_list:
        if (item.endswith(".txt")):
            os.remove(os.path.join(dir_path, item))