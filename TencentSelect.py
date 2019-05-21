import re


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # 2.两数相加
    # 给出两个非空的链表用来表示两个非负的整数。其中，它们各自的位数是按照逆序的方式存储的，并且它们的每个节点只能存储一位数字。
    @staticmethod
    def add_two_numbers(l1: ListNode, l2: ListNode) -> ListNode:
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
    def find_median_sorted_arrays(nums1: [int], nums2: [int]) -> float:
        nums = nums1 + nums2
        nums.sort()
        return float(nums[len(nums) // 2]) if len(nums) % 2 == 1 \
            else (nums[len(nums) // 2 - 1] + nums[len(nums) // 2]) / 2

    # 5.最长回文子串
    @staticmethod
    def longest_palindrome(s: str) -> str:
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

    # 7.整数反转
    @staticmethod
    def reverse(x: int) -> int:
        s = str(abs(x))
        if (int(s[::-1]) > 2 * 31 - 1 and x > 0) or (int(s[::-1]) > 2 * 31 and x < 0):
            return 0
        else:
            return int(s[::-1]) if x > 0 else -int(s[::-1])

    # 8.字符串转换整数（atoi）
    @staticmethod
    def my_atoi(s: str) -> int:
        s = s.strip()
        rg = '(^[\+\-0]*\d+)\D*'
        s = re.findall(rg, s)

        try:
            result = int(''.join(s))
            if result > 2 * 31 - 1 > 0:
                return 2 * 31 - 1
            elif result < - 2 * 31 < 0:
                return - 2 * 31
            else:
                return result
        except:
            return 0

    # 9.回文数
    @staticmethod
    def is_palindrome(x: int) -> bool:
        if x < 0:
            return False
        h = x
        temp = 0
        while h:
            temp = temp * 10 + h % 10
            h //= 10
        return temp == x

    # 11.盛最多水的容器
    @staticmethod
    def max_area(height: [int]) -> int:
        i = 0
        j = len(height) - 1
        res = 0
        while i < j:
            res = max(res, min(height[i], height[j]) * (j - i))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return res

    # 14.最长公共前缀
    @staticmethod
    def longest_common_prefix(strs: [str]) -> str:
        if not strs:
            return ''
        shortest = min(strs, key=len)
        for i, ch in enumerate(shortest):
            for s in strs:
                if shortest[i] != s[i]:
                    return shortest[:i]
        return shortest

        # res = ''
        # if not strs:
        #     return ''
        # for each in zip(*strs):
        #     if len(set(each)) == 1:
        #         res += each[0]
        #     else:
        #         return res
        # return res

    # 15.三数之和
    @staticmethod
    def three_sum(nums: [int]) -> [[int]]:
        res = []
        nums.sort()
        for i in range(len(nums)):
            if i == 0 or nums[i] > nums[i - 1]:
                left = i + 1
                right = len(nums) - 1
                while left < right:
                    val = nums[i] + nums[left] + nums[right]
                    if val == 0:
                        res.append([nums[i], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        while left < right and nums[left] == nums[left - 1]:
                            left += 1
                        while left < right and nums[right] == nums[right + 1]:
                            right -= 1
                    elif val < 0:
                        left += 1
                    else:
                        right -= 1
        return res

    # 16.最接近的三数之和
    @staticmethod
    def three_sum_closet(nums: [int], target: int) -> int:
        nums.sort()
        res = []

        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left = i + 1
            right = len(nums) - 1
            if nums[i] + nums[right - 1] + nums[right] < target:
                res.append(nums[i] + nums[right - 1] + nums[right])
            elif nums[i] + nums[left] + nums[left + 1] > target:
                res.append(nums[i] + nums[left] + nums[left + 1])
            else:
                while left < right:
                    s = nums[i ] + nums[left] + nums[right]
                    res.append(s)
                    if s == target:
                        return target
                    elif s < target:
                        left += 1
                    else:
                        right -= 1
            res.sort(key=lambda x: abs(x - target))
            return res[0]

    # 20.有效的括号
    @staticmethod
    def is_valid(s: str) -> bool:
        pre_dict = {')': '(', ']': '[', '}': '{'}
        stack = []
        for i in s:
            if i in pre_dict.values():
                stack.append(i)
            elif not stack or pre_dict[i] != stack.pop():
                return False
        return not stack

    