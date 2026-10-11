from .Tool import Tool
import os
import subprocess
from typing import List,Any,Dict
from pathlib import Path
from .ToolParameter import ToolParameter


class TerminalTool(Tool):
    def __init__(self,
                 name: str,
                 description: str,
                 timeout: int,
                 max_output_size: int,
                 current_dir: Path,
                 workspace: Path,
                 allow_cd: bool):
        super().__init__(name, description)
        self.timeout = timeout
        self.max_output_size = max_output_size
        # 强制转为Path对象，做类型约束
        self.workspace: Path = workspace.resolve()
        self.current_dir: Path = current_dir.resolve()
        self.allow_cd = allow_cd

        # 初始化校验：初始目录必须在workspace沙箱内
        try:
            self.current_dir.relative_to(self.workspace)
        except ValueError:
            raise ValueError(f"初始工作目录 {self.current_dir} 不能超出沙箱workspace {self.workspace}")

    def _execute_command(self, command: str) -> str:
        """执行命令"""
        try:
            # 在当前目录下执行命令
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.current_dir),  # 在当前工作目录执行
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=os.environ.copy()
            )

            # 合并标准输出和标准错误
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            # 检查输出大小
            if len(output) > self.max_output_size:
                output = output[:self.max_output_size]
                output += f"\n\n⚠️ 输出被截断（超过 {self.max_output_size} 字节）"

            # 添加返回码信息
            if result.returncode != 0:
                output = f"⚠️ 命令返回码: {result.returncode}\n\n{output}"

            return output if output else "✅ 命令执行成功（无输出）"

        except subprocess.TimeoutExpired:
            return f"❌ 命令执行超时（超过 {self.timeout} 秒）"
        except Exception as e:
            return f"❌ 命令执行失败: {e}"

    def _handle_cd(self, parts: List[str]) -> str:
        """处理 cd 命令"""
        if not self.allow_cd:
            return "❌ cd 命令已禁用"

        if len(parts) < 2:
            # cd 无参数，返回当前目录
            return f"当前目录: {self.current_dir}"

        target_dir = parts[1]

        # 处理相对路径
        if target_dir == "..":
            new_dir = self.current_dir.parent
        elif target_dir == ".":
            new_dir = self.current_dir
        elif target_dir == "~":
            new_dir = self.workspace
        else:
            new_dir = (self.current_dir / target_dir).resolve()

        # 检查是否在工作目录内
        try:
            new_dir.relative_to(self.workspace)
        except ValueError:
            return f"❌ 不允许访问工作目录外的路径: {new_dir}"

        # 检查目录是否存在
        if not new_dir.exists():
            return f"❌ 目录不存在: {new_dir}"

        if not new_dir.is_dir():
            return f"❌ 不是目录: {new_dir}"

        # 更新当前目录
        self.current_dir = new_dir
        return f"✅ 切换到目录: {self.current_dir}"

    def execute(self, parameters: Dict[str, Any]) -> Any:
        """
        实现抽象基类Tool的execute，function‑call调用入口
        parameters: {"command": "cd ../src"}
        """
        command = parameters.get("command", "")
        if not command:
            return "❌ 参数错误，缺少 command"

        # 简单切分命令
        parts = command.strip().split()
        if not parts:
            return "❌ 空命令"

        cmd_head = parts[0].lower()
        # 分发内置命令
        if cmd_head == "cd":
            return self._handle_cd(parts)
        else:
            # 其余命令交给subprocess执行
            return self._execute_command(command)

    def get_parameters(self) -> List[ToolParameter]:
        """实现抽象方法：返回工具参数定义，用于to_openai_schema"""
        return [
            ToolParameter(
                name="command",
                type="string",
                description="要执行的shell命令，支持cd、ls等；cd为内置命令，会维护工具内部工作目录",
                required=True
            )
        ]