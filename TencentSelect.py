import re
import math


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

    # 21.合并两个有序链表
    # 法一：递归
    def merge_two_lists1(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val > l2.val:
            l1, l2 = l2, l1
        l1.next = self.merge_two_lists1(l1.next, l2)
        return l1

    # 法二：非递归
    @staticmethod
    def merge_two_lists2(l1: ListNode, l2: ListNode) -> ListNode:
        res = ListNode(0)
        node = res
        while l1 and l2:
            if l1.val < l2.val:
                node.next, l1 = l1, l1.next
            else:
                node.next, l2 = l2, l2.next
            node = node.next
        if l1:
            node.next = l1
        if l2:
            node.next = l2
        return res.next

    # 23.合并K个排序表
    @staticmethod
    def merge_k_lists(lists: [ListNode]) -> ListNode:
        temp = []
        for i in lists:
            while i:
                temp.append(i)
                i = i.next
        h = res = ListNode(0)
        for node in sorted(temp, key=lambda n: n.val):
            h.next = node
            h = h.next
        return res.next

    # 26.删除排序数组中的重复项
    # 快慢指针法，也可以使用倒序pop
    @staticmethod
    def remove_duplicates(nums: [int]) -> int:
        i = 0
        for j in range(1, len(nums)):
            if nums[i] != nums[j]:
                nums[i + 1] = nums[j]
        return i + 1 if nums else 0

    # 33.搜索旋转排序数组
    @staticmethod
    def search(nums: [int], target: int) -> int:
        return nums.index(target) if target in nums else -1

        # keys = [i for i in range(0, len(nums))]
        # dictTemp = dict(zip(nums, keys))
        # return dictTemp[target] if target in dictTemp.keys() else -1

    # 43.字符串相乘

    # 46.全排列
    def permute(self, nums: [int]) -> [[int]]:
        if len(nums) <= 1:
            return [nums]
        res = []
        for i in range(len(nums)):
            temp = self.permute(nums[:i] + nums[i + 1:])
            for j in temp:
                res.append([nums[i]] + j)
        return res

    # 53.最大子序和
    @staticmethod
    def max_sub_array(nums: [int]) -> int:
        cur = res = nums[0]
        for num in nums[1:]:
            cur, res = max(cur + num, num), max(res, cur)
        return res

    # 54.螺旋矩阵
    # 法一：递归
    def spiral_order1(self, matrix: [[int]]) -> [int]:
        if not matrix:
            return []
        res = []
        res.extend(matrix[0])
        new = [reversed(i) for i in matrix[1:]]
        if not new:
            return res
        temp = self.spiral_order1([i for i in zip(*new)])
        res.extend(temp)
        return res

    # 法二
    @staticmethod
    def spiral_order2(matrix: [[int]]) -> [int]:
        if not matrix:
            return []
        res = []
        m, n = len(matrix), len(matrix[0])
        c, j = 0, 0 # c为总数count，j为计数count
        while c < m * n:
            for i in range(j, n - j):
                res.append(matrix[j][i])
                c += 1
            for i in range(j + 1, m - j):
                res.append(matrix[i][n - j - 1])
                c += 1
            for i in range(n - j - 2, j - 1, -1):
                res.append(matrix[m - j -1][i])
                c += 1
            for i in range(m - j - 2, j, -1):
                res.append(matrix[i][j])
                c += 1
            j += 1
        return res[:m * n]

    # 59.螺旋矩阵II
    @staticmethod
    def generate_matrix(n: int) -> [[int]]:
        if n <= 0:
            return [[]]
        matrix = [[0] * n for _ in range(n)]
        num = 1
        c, j = n ** 2, 0
        while num <= c:
            for i in range(j, n - j):
                matrix[j][i] = num
                num += 1
                if num > c:
                    return matrix
            for i in range(j + 1, n - j):
                matrix[i][n - j - 1] = num
                num += 1
            for i in range(n - j - 2, j - 1, -1):
                matrix[n - j - 1][i] = num
                num += 1
                if num > c:
                    return matrix
            for i in range(n - j - 2, j, -1):
                matrix[i][j] = num
                num += 1
            j += 1
        return matrix

    # 61.旋转链表
    @staticmethod
    def rotate_right(head: ListNode, k: int) -> ListNode:
        if k == 0 or not head or not head.next:
            return head
        h = head
        n = 1
        while h.next:
            h = h.next
            n += 1
        if k % n == 0:
            return head
        h.next = head
        h = h.next
        m = n - k % n
        for _ in range(m - 1):
            h = h.next
        new_h = h.next
        h.next = None
        return new_h

    # 62.不同路径
    @staticmethod
    def unique_path(m: int, n: int) -> int:
        return int(math.factorial(m + n - 2) / math.factorial(m - 1) / math.factorial(n - 1))

