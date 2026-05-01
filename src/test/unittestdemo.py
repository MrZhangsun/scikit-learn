import unittest

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b
class TestAdd(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(1, 2), 3)


    def test_sub(self):
        self.assertEqual(sub(1, 2), -1)

    def test_mul(self):
        self.assertEqual(mul(1, 2), 2)

if __name__ == "__main__":
    unittest.main()