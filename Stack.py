import collections


class TreeNodes:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    # 20
    @staticmethod
    def isvalid(self, s) -> bool:
        pre_dict = {')': '(', ']': '[', '}': '{'}
        stack = []
        for i in s:
            if i in pre_dict.values():
                stack.append(i)
            elif not stack or pre_dict[i] != stack.pop():
                return False
        return not stack

    # 42
    # 接雨水
    # 给定n个非负整数表示每个宽度为1的柱子的高度图，计算按此排列的柱子，下雨之后能接多少水
    # 法一
    @staticmethod
    def trap1(self, height: [int]) -> int:
        if not height:
            return 0

        max_height = 0
        max_height_index = 0

        # 找到最高点
        for i in range(len(height)):
            h = height[i]
            if h > max_height:
                max_height = h
                max_height_index = i

        area = 0

        # 分别从左和右往最高点遍历
        temp = height[0]
        for i in range(max_height_index):
            if height[i] > temp:
                temp = height[i]
            else:
                area += (temp - height[i])

        temp = height[-1]
        for i in reversed(range(max_height_index + 1, len(height))):
            if height[i] > temp:
                temp = height[i]
            else:
                area += (temp - height[i])

        return area

    # 法二
    @staticmethod
    def trap2(self, height: [int]) -> int:
        minHeight = area = 0
        start, end = 0, len(height) - 1
        while start < end:
            while start < end and height[start] <= minHeight:
                area += minHeight - height[start]
                start += 1
            while start < end and height[end] <= minHeight:
                area += minHeight - height[end]
                end -= 1
            minHeight = min(height[start], height[end])
        return area

    # 71 Unix路径简化
    @staticmethod
    def simplify_path(self, path: str) -> str:
        res = []
        temp = path.split('/')
        for i in temp:
            if i == '' or i == '.':
                continue
            elif i == '..':
                if res:
                    res.pop()
            else:
                res.append(i)
        return '/' + '/'.join(res)

    # 84 柱状图中最大矩形
    # 法一
    @staticmethod
    def largest_rectangle_area1(self, heights: [int]) -> int:
        i = 0
        max_value = 0
        stack = []
        heights.append(0)

        while i < len(heights):
            if not stack or heights[stack[-1]] <= heights[i]:
                stack.append(i)
                i += 1
            else:
                now_index = stack.pop()
                if not stack:
                    max_value = max(max_value, i * heights[now_index])
                else:
                    max_value = max(max_value, (i - stack[-1] - 1) * heights[now_index])

        return max_value

    # 法二
    @staticmethod
    def largest_rectangle_area2(self, heights: [int]) -> int:
        stack = []
        area = 0
        for i in range(len(heights) + 1):
            cur = heights[i] if i != len(heights) else -1
            while stack and heights[stack[-1]] > cur:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1 if stack else i
                area = max(area, w * h)
            stack.append(i)
        return area

    # 84 最大矩形
    # 给定一个仅包含0和1的二维二进制矩阵，找出只包含1的最大矩形，并返回其面积
    @staticmethod
    def maximal_rectangle(self, matrix: [[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        nums = [int(''.join(row), base=2) for row in matrix]
        ans, N = 0, len(nums)
        for i in range(N):
            j, num = i, nums[i]
            while j < N:
                num &= nums[j]
                if not num:
                    break
                l, curnum = 0, num
                while curnum:
                    l += 1
                    curnum &= (curnum << 1)
                ans = max(ans, l * (j - i + 1))
                j += 1
        return ans

    # 94 二叉树的中序遍历
    # 法一（栈）
    @staticmethod
    def inorder_traversal1(self, root: TreeNodes) -> [int]:
        res = []
        stack = []
        p = root
        while p or stack:
            while p:
                stack.append(p)
                p = p.left
            if stack:
                p = stack.pop()
                res.append(p.val)
                p = p.right
        return res

    # 法二（递归）
    def inorder_traversal2(self, root: TreeNodes) -> [int]:
        if not root:
            return []
        res = []
        if root.left:
            res += self.inorderTraversal2(root.left)
        res.append(root.val)
        if root.right:
            res += self.inorderTraversal2(root.right)
        return res

    # 103 二叉树的锯齿形层次遍历
    # 给定一个二叉树，返回其节点值得锯齿形层次遍历（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）
    @staticmethod
    def zigzag_level_order(self, root: TreeNodes) -> [[int]]:
        if not root:
            return []
        res = []
        stack = []
        flag = -1
        stack.append(root)
        while stack:
            flag = -flag
            val = []
            temp = []
            for i in stack:
                val.append(i.val)
                if i.left:
                    temp.append(i.left)
                if i.right:
                    temp.append(i.right)
            res.append(val[::flag])
            stack = temp
        return res

    # 144 二叉树的前序遍历
    # 法一（栈）
    @staticmethod
    def preorder_traversal1(self, root: TreeNodes) -> [int]:
        if not root:
            return []
        res = []
        stack = [root]
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

    # 法二（递归）
    def preorder_traversal2(self, root: TreeNodes) -> [int]:
        if not root:
            return []
        res = []
        res.append(root.val)
        if root.left:
            res += self.preorderTraversal2(root.left)
        if root.right:
            res += self.preorderTraversal2(root.right)
        return res

    # 145 二叉树的后序遍历
    # 法一（栈）
    @staticmethod
    def postorder_traversal1(self, root: TreeNodes) -> [int]:
        if not root:
            return []
        stack = [root]
        temp = []
        while stack:
            node = stack.pop()
            temp.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return temp[::-1]

    # 法二（递归）
    def postorder_traversal2(self, root: TreeNodes) -> [int]:
        if not root:
            return []
        res = []
        if root.left:
            res += self.postorderTraversal2(root.left)
        if root.right:
            res += self.postorderTraversal2(root.right)
        res.append(root.val)
        return res

    # 150 逆波兰表达式求值
    @staticmethod
    def evalRPN(self, tokens: [str]) -> int:
        stack = []
        for i in tokens:
            if i in ("+", "-", "*", "/"):
                stack.append(i)
            else:
                int2 = int(stack.pop())
                int1 = int(stack.pop())
                if i == "+":
                    stack.append(str(int1 + int2))
                elif i == "*":
                    stack.append(str(int1 * int2))
                elif i == "-":
                    stack.append(str(int1 - int2))
                else:
                    stack.append(str(int(int1 / int2)))
        return int(stack.pop())

    # 224 基本计算器
    # 实现一个基本的计算器来计算一个简单的字符串表达式的值
    # 字符串表达式可以包含左括号，右括号，加号，减号，非负整数和空格
    @staticmethod
    def calculate(self, s: str) -> int:
        res, num, sign = 0, 0, 1
        stack = []
        for c in s:
            if c.isdigit():
                num = 10 * num + c
            elif c == "+" or c == "-":
                res += sign * num
                num = 0
                sign = 1 if c == "+" else -1
            elif c == "(":
                stack.append(res)
                stack.append(sign)
                res = 0
                sign = 1
            elif c == ")":
                res += sign * num
                num = 0
                res *= stack.pop()
                res += stack.pop()
        res += sign * num
        return res

    # 316 去除重复字母
    # 给定一个仅包含小写字母的字符串，去除字符串中重复的字母，使得每个字母只出现一次。
    # 需保证返回结果的字典序最小（要求不能打乱其他字符的相对位置）
    @staticmethod
    def remove_duplicate_letters(s: str) -> str:
        count = collections.Counter(s)
        stack = []
        visited = collections.defaultdict(bool)
        for c in s:
            count[c] -= 1
            if visited[c]:
                continue
            while stack and count[stack[-1]] and stack[-1] > c:
                visited[stack[-1]] = False
                stack.pop()
            visited[c] = True
            stack.append(c)
        return "".join(stack)


# 173 二叉搜索树迭代器
class BSTIterator:

    def __init__(self, root: TreeNodes):
        self.stack = []
        self.push_left(root)

    def next(self) -> int:
        """
        :return: the next smallest number
        """
        node = self.stack.pop()
        self.push_left(node.right)
        return node.val

    def has_next(self) -> bool:
        """
        :return: whether we have a next smallest number
        """
        return True if self.stack else False

    def push_left(self, root):
        while root:
            self.stack.append(root)
            root = root.left


