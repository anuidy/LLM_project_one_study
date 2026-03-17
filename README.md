# LLM_project_one_study
大模型项目-1

## 仓库说明

本仓库用于记录 MedicalGPT 项目的学习过程，包含以下文档：

| 文件 | 说明 |
|------|------|
| [`docs/medicalgpt-7day-study-plan.md`](docs/medicalgpt-7day-study-plan.md) | MedicalGPT 7 天学习计划与学习清单 |
| [`docs/medicalgpt-2x4090-command-checklist.md`](docs/medicalgpt-2x4090-command-checklist.md) | 双 RTX 4090 可执行命令清单 |

## 如何查看新增文档

### 方式一：直接在 GitHub 网页上查看（无需任何操作）

1. 打开仓库主页：`https://github.com/anuidy/LLM_project_one_study`
2. 点击页面左上角的分支下拉菜单（默认显示 `main`）
3. 切换到分支：`copilot/add-docs-medicalgpt-study`
4. 点击 `docs/` 目录，即可看到两份新文档

### 方式二：合并 Pull Request（推荐，让文档永久留在 main 分支）

1. 打开 PR 页面：[PR #1](https://github.com/anuidy/LLM_project_one_study/pull/1)
2. 页面底部点击绿色按钮 **"Ready for review"**（将草稿转为正式 PR）
3. 再点击 **"Merge pull request"** → **"Confirm merge"**
4. 合并完成后，回到 `main` 分支，`docs/` 目录下即可看到两份文档

## 不希望将问答/对话上传到仓库，应该怎么设置？

如果你希望查看新增文档时做的问答记录，但**不提交到 GitHub 仓库**，可以按下面方式处理：

1. 在仓库本地创建目录（示例）：`qa_logs/` 或 `private_notes/`
2. 把问答内容保存到这些目录里，或保存为 `*.conversation.md`、`*.conversation.txt` 或 `*.conversation.json` 文件
3. 仓库已添加 `.gitignore` 规则，这些内容默认不会被提交

可用下面命令自检（如果输出中仍出现问答记录文件，说明需要再补充忽略规则）：

```bash
git status --short
```
