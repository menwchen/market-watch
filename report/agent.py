"""ReACT pattern report agent using Claude API with real market data tools."""

import json
import re
from datetime import datetime
from typing import Optional

import anthropic

from config import Config
from report.tools import ReportTools
from report.templates import (
    PLAN_SYSTEM_PROMPT,
    PLAN_USER_PROMPT,
    SECTION_SYSTEM_PROMPT,
    SECTION_USER_PROMPT,
)
from storage.reports import ReportStore


class ReportAgent:
    """Generates market analysis reports using ReACT pattern with real data."""

    def __init__(self, language: str = "Korean"):
        self.client = anthropic.Anthropic(
            api_key=Config.ANTHROPIC_API_KEY,
            timeout=120.0,
        )
        self.model = Config.LLM_MODEL
        self.tools = ReportTools()
        self.store = ReportStore()
        self.language = language
        self.log: list[dict] = []
        self.max_sections = 3  # Keep sections low for memory/time

    def _call_llm(self, system: str, messages: list[dict],
                  max_tokens: int = 2048) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    def _log(self, event: str, data: dict):
        entry = {"timestamp": datetime.now().isoformat(), "event": event, **data}
        self.log.append(entry)
        # Print progress to console
        if event == "section_start":
            print(f"\n{'='*60}")
            print(f"  Generating: {data.get('title', '')}")
            print(f"{'='*60}")
        elif event == "tool_call":
            print(f"  -> Tool: {data.get('tool', '')}({data.get('args', '')})")
        elif event == "section_complete":
            print(f"  [OK] Section complete ({data.get('length', 0)} chars)")

    def generate_report(self, assets: list[str], period: str = "3mo") -> str:
        """Full report generation pipeline."""
        print("\n[1/3] Collecting market data snapshot...")
        snapshot = self._collect_snapshot(assets, period)

        print("\n[2/3] Planning report structure...")
        outline = self._plan_outline(assets, period, snapshot)

        print(f"\n[3/3] Generating {len(outline['sections'])} sections...")
        sections = []
        for i, section_def in enumerate(outline["sections"]):
            content = self._generate_section(
                section_def, outline, assets, period, i + 1
            )
            sections.append({"title": section_def["title"], "content": content})

        report = self._assemble_report(outline, sections)

        filepath = self.store.save_report(
            content=report,
            title=outline["title"],
            metadata={
                "assets": assets,
                "period": period,
                "sections": len(sections),
                "language": self.language,
                "log_entries": len(self.log),
            },
        )
        print(f"\nReport saved: {filepath}")
        return report

    def _collect_snapshot(self, assets: list[str], period: str) -> str:
        lines = []
        for asset in assets:
            data = self.tools.get_price_data(asset=asset, period=period)
            if "error" not in data:
                price = data.get("price", "N/A")
                change_pct = data.get("change_pct", 0)
                period_ret = data.get("period_return", 0)
                lines.append(
                    f"- {asset}: ${price} ({change_pct:+.2f}% today, "
                    f"{period_ret:+.1f}% over period)"
                )
            else:
                lines.append(f"- {asset}: {data.get('error', 'No data')}")
        return "\n".join(lines)

    def _plan_outline(self, assets: list[str], period: str,
                      snapshot: str) -> dict:
        prompt = PLAN_USER_PROMPT.format(
            assets=", ".join(assets),
            period=period,
            snapshot=snapshot,
        )
        response = self._call_llm(
            system=PLAN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            outline = json.loads(json_match.group())
        else:
            outline = {
                "title": "Market Analysis Report",
                "summary": "Comprehensive market analysis",
                "sections": [
                    {"title": "Market Overview", "description": "Current market conditions"},
                    {"title": "Technical Analysis", "description": "Technical indicators"},
                    {"title": "Forecast & Risk", "description": "Forward-looking analysis"},
                ],
            }
        # Limit sections to save time/memory
        if len(outline.get("sections", [])) > self.max_sections:
            outline["sections"] = outline["sections"][:self.max_sections]
        self._log("outline", {"title": outline["title"],
                               "sections": len(outline["sections"])})
        return outline

    def _generate_section(self, section_def: dict, outline: dict,
                          assets: list[str], period: str,
                          section_num: int) -> str:
        self._log("section_start", {"title": section_def["title"], "num": section_num})

        system = SECTION_SYSTEM_PROMPT.format(
            tool_descriptions=self.tools.get_tool_descriptions(),
            language=self.language,
        )
        user_msg = SECTION_USER_PROMPT.format(
            section_title=section_def["title"],
            section_description=section_def.get("description", ""),
            report_context=f"Report: {outline['title']}\n"
                           f"Assets: {', '.join(assets)}\nPeriod: {period}",
        )

        messages = [{"role": "user", "content": user_msg}]
        tool_calls_made = 0
        max_iterations = Config.REPORT_MAX_TOOL_CALLS

        for _ in range(max_iterations):
            response = self._call_llm(system=system, messages=messages)

            # Check for Final Answer
            if "Final Answer:" in response:
                final = response.split("Final Answer:")[-1].strip()
                self._log("section_complete",
                          {"title": section_def["title"], "length": len(final)})
                return final

            # Parse Action from response
            action_match = re.search(
                r'Action:\s*(\w+)\(([^)]*)\)', response
            )
            if action_match:
                tool_name = action_match.group(1)
                args_str = action_match.group(2).strip()
                arguments = self._parse_tool_args(tool_name, args_str)

                self._log("tool_call", {"tool": tool_name, "args": arguments})
                tool_result = self.tools.execute(tool_name, arguments)
                tool_calls_made += 1

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {tool_result}\n\n"
                               f"Continue with your next Thought, or write "
                               f"'Final Answer:' followed by the section content.",
                })
            else:
                # No action found - treat response as final answer
                self._log("section_complete",
                          {"title": section_def["title"], "length": len(response)})
                return response

        # Max iterations reached - ask for final answer
        messages.append({
            "role": "user",
            "content": "You have used all available tool calls. "
                       "Write 'Final Answer:' followed by the section content now.",
        })
        response = self._call_llm(system=system, messages=messages)
        final = response.split("Final Answer:")[-1].strip() if "Final Answer:" in response else response
        self._log("section_complete",
                  {"title": section_def["title"], "length": len(final)})
        return final

    def _parse_tool_args(self, tool_name: str, args_str: str) -> dict:
        """Parse tool arguments from the LLM's Action string."""
        if not args_str:
            return {}

        # Try JSON-like parsing first
        args_str = args_str.strip()
        if args_str.startswith("{"):
            try:
                return json.loads(args_str)
            except json.JSONDecodeError:
                pass

        # Simple key=value parsing
        args = {}
        # Find the tool definition to know expected params
        tool_def = next(
            (t for t in ReportTools.TOOL_DEFINITIONS if t["name"] == tool_name),
            None,
        )
        if not tool_def:
            return {}

        param_names = list(tool_def.get("parameters", {}).keys())

        # Handle positional args
        parts = [p.strip().strip("'\"") for p in args_str.split(",")]

        for i, part in enumerate(parts):
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if val.isdigit():
                    val = int(val)
                args[key] = val
            elif i < len(param_names):
                val = part.strip("'\"")
                if val.isdigit():
                    val = int(val)
                args[param_names[i]] = val

        return args

    def _assemble_report(self, outline: dict, sections: list[dict]) -> str:
        lines = [
            f"# {outline['title']}",
            f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
            f"**{outline.get('summary', '')}**\n",
            "---\n",
        ]
        for i, section in enumerate(sections, 1):
            lines.append(f"## {i}. {section['title']}\n")
            lines.append(section["content"])
            lines.append("\n---\n")

        lines.append(
            f"\n*This report was generated by MarketPulse using real market data "
            f"from Yahoo Finance, FRED, and EIA APIs. "
            f"Past performance does not guarantee future results.*\n"
        )
        return "\n".join(lines)
