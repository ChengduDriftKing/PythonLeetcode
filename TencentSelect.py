import re
import math
import itertools
import collections


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

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

    # 70.爬楼梯
    # 第n阶台阶的走法可以看做是1步加上n-1阶台阶的走法与2步加上n-2阶台阶的走法，所以可以转化为斐波那契数列
    # dp[n] = dp[n - 1] + dp[n - 2]
    @staticmethod
    def climb_stairs(n: int) -> int:
        p, q = 1, 2
        if n == 1:
            return 1
        if n == 2:
            return 2
        for _ in range(2, n):
            p, q = q, p+q
        return q

    # 78.子集
    # 给定一组不含重复元素的整数数组nums，返回该数组所有可能的子集（幂集）
    # 法一：
    @staticmethod
    def subsets1(nums: [int]) -> [[int]]:
        res = [[]]
        for num in nums:
            for temp in res[:]:
                x = temp[:]
                x.append(num)
                res.append(x)
        return res

    # 法二：回溯法递归
    @staticmethod
    def subsets2(nums: [int]) -> [[int]]:
        res = []

        def helper(i, tmp):
            res.append(tmp)
            for j in range(i, len(nums)):
                helper(j + 1, tmp + [nums[j]])
        helper(0, [])
        return res

    # 法三：
    @staticmethod
    def subsets3(nums: [int]) -> [[int]]:
        res = [[]]
        for i in nums:
            res += [[i] + num for num in res]
        return res

    # 法四：
    @staticmethod
    def subsets4(nums: [int]) -> [[int]]:
        res = []
        for i in range(len(nums) + 1):
            for tmp in itertools.combinations(nums, i):
                res.append(tmp)
        return res

    # 88 合并两个有序数组
    @staticmethod
    def merge(nums1: [int], m: int, nums2: [int], n: int) -> None:
        while n > 0:
            if m and nums1[m - 1] > nums2[n - 1]:
                nums1[m + n - 1], m = nums1[m - 1], m - 1
            else:
                nums1[m + n - 1], n = nums2[n - 1], n - 1

    # 89 格雷编码
    @staticmethod
    def gray_code(n: int) -> [int]:
        return [i ^ i >> 1 for i in range(1 << n)]

    # 104 二叉树的最大深度
    # 法一 DFS（深度优先算法）：
    def max_depth1(self, root: TreeNode) -> int:
        if not root:
            return 0
        left_height = self.max_depth1(root.left)
        right_height = self.max_depth1(root.right)
        return max(left_height, right_height) + 1

    # 法二 迭代法：
    @staticmethod
    def max_depth2(root: TreeNode) -> int:
        stack = []
        if root:
            stack.append((1, root))

        depth = 0
        while stack:
            current_depth, root = stack.pop()
            if root:
                depth = max(depth, current_depth)
                stack.append((current_depth + 1, root.left))
                stack.append((current_depth + 1, root.right))
        return depth

    # 121 买卖股票的最佳时机
    @staticmethod
    def max_profit(prices: [int]) -> int:
        n = len(prices)
        dp_i_0, dp_i_1 = 0, -float("inf")
        for i in range(n):
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i])
            dp_i_1 = max(dp_i_1, -prices[i])
        return dp_i_0

    # 122 买卖股票的最佳时机2
    @staticmethod
    def max_profit_second(prices: [int]) -> int:
        dp_i_0, dp_i_1 = 0, -float("inf")
        for i in range(len(prices)):
            temp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i])
            dp_i_1 = max(dp_i_1, temp - prices[i])
        return dp_i_0

    # 124 二叉树中的最大路径和
    def max_path_sum(self, root: TreeNode) -> int:
        self.res = -float("inf")
        self.max_gain(root)
        return self.res

    def max_gain(self, node: TreeNode) -> int:
        if not node:
            return 0
        left_gain = max(self.max_gain(node.left), 0)
        right_gain = max(self.max_gain(node.right), 0)
        self.res = max(self.res, node.val + left_gain + right_gain)
        return node.val + max(left_gain, right_gain)

    # 136 只出现一次的数字
    # 法一 数学法：
    @staticmethod
    def single_number1(nums: [int]) -> int:
        return 2 * sum(set(nums)) - sum(nums)

    # 法二 位操作：
    @staticmethod
    def single_number2(nums: [int]) -> int:
        res = 0
        for i in nums:
            res ^= i
        return res

    # 141 环形链表
    # 法一 双指针
    @staticmethod
    def has_cycle1(head: ListNode) -> bool:
        slow = fast = head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False

    # 法二 set
    @staticmethod
    def has_cycle2(head: ListNode) -> bool:
        res = set()
        while head:
            if head in res:
                return True
            res.add(head)
            head = head.next
        return False

    # 142 环形链表2
    @staticmethod
    def detect_cycle(head: ListNode) -> ListNode:
        res = set()
        while head:
            if head in res:
                return head
            res.add(head)
            head = head.next
        return None

    # 148 排序链表
    @staticmethod
    def sort_list(head: ListNode) -> ListNode:
        tmp = []
        res = res_tmp = head
        while head:
            tmp.append(head.val)
            head = head.next
        tmp.sort()
        for i in tmp:
            res_tmp.val = i
            res_tmp = res_tmp.next
        return res

    # 160 相交链表
    @staticmethod
    def get_intersection_node(headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None
        pA, pB = headA, headB
        while pA != pB:
            pA = headB if not pA else pA.next
            pB = headA if not pB else pB.next
        return pA

    # 169 求众数
    # 法一 数学法
    @staticmethod
    def majority_element1(nums: [int]) -> int:
        nums.sort()
        return nums[len(nums) // 2]

    # 法二 字典
    @staticmethod
    def majority_element2(nums: [int]) -> int:
        dic = {}
        for i in nums:
            if i not in dic.keys():
                dic[i] = 0
            dic[i] += 1
            if dic[i] >= len(nums) / 2:
                return i

    # 206 反转链表
    # 法一 迭代法
    @staticmethod
    def reverse_list1(head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur:
            cur.next, cur, pre = pre, cur.next, cur
        return pre

    # 法二 递归
    def reverse_list2(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        p = self.reverse_list2(head.next)
        head.next.next = head
        head.next = None
        return p

    


# 146 LRU缓存机制
class LRUCache:
    def __init__(self, capacity: int):
        self.remain = capacity
        self.dic = collections.OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.dic:
            return -1
        self.dic.move_to_end(key)
        return self.dic[key]

    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            self.dic.pop(key)
        else:
            if self.remain > 0:
                self.remain -= 1
            else:
                self.dic.popitem(last=False)
        self.dic[key] = value

