# coding=utf-8

import csv
import os
import sys


class CategoryHelper(object):
    """ category helper class : load categories and get category name by id"""

    def __init__(self, fcategory):
        self._category_name = []

        if not os.path.exists(fcategory):
            raise IOError("File is not exists: " + fcategory)
        with open(fcategory) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'CategoryId':
                    continue
                self._category_name.append(row[1])
        print("got categories count: " + str(len(self._category_name)))

    def get_category_names(self):
        return self._category_name
        
    def get_category_name(self, category_id):
        """
        :param category_id: must start by 1
        :return: category name
        """
        if category_id < 1 or category_id > len(self._category_name):
            return "None"
        return self._category_name[category_id - 1]


def test(category_id):
    helper = CategoryHelper("categories.csv")

    print(str(category_id) + "\t" + helper.get_category_name(category_id))


if __name__ == "__main__":
    """python category.py 105"""
    if len(sys.argv) < 2:
        print("python category.py category_id")
        print("example: python category.py 105")
        exit(1)

    test(int(sys.argv[1]))
