class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # 2.两数相加
    # 给出两个非空的链表用来表示两个非负的整数。其中，它们各自的位数是按照逆序的方式存储的，并且它们的每个节点只能存储一位数字。
    @staticmethod
    def add_two_numbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        num1 = num2 = 0
        base1 = base2 = 1
        while l1:
            num1 += l1.val * base1
            base1 *= 10
            l1 = l1.next
        while l2:
            num2 += l2.val * base2
            base2 *= 10
            l2 = l2.next
        num = num1 + num2
        res = ListNode(0)
        if num == 0:
            return res
        h = res
        while num != 0:
            h.next = ListNode(num % 10)
            h = h.next
            num //= 10
        return res.next


    # 4.寻找两个有序数组的中位数
    @staticmethod
    def find_median_sorted_arrays(self, nums1: [int], nums2: [int]) -> float:
        nums = nums1 + nums2
        nums.sort()
        return float(nums[len(nums) // 2]) if len(nums) % 2 == 1 \
            else (nums[len(nums) // 2 - 1] + nums[len(nums) // 2]) / 2


    # 5.最长回文子串
    @staticmethod
    def longest_palindrome(self, s: str) -> str:
        if len(s) < 2 or s == s[::-1]:
            return s
        start, maxLen = 0, 1
        for i in range(1, len(s)):
            odd = s[i - maxLen - 1: i + 1]
            even = s[i - maxLen: i + 1]

            if i - maxLen - 1 >= 0 and odd == odd[::-1]:
                start = i - maxLen - 1
                maxLen += 2
                continue
            if i - maxLen >= 0 and even == even[::-1]:
                start = i - maxLen
                maxLen += 1

        return s[start: start + maxLen]

