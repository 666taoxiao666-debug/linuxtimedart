from __future__ import annotations

import argparse
import random
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
SPRITE_PATH = ROOT / "assets" / "girlfriend_pet.png"
TRANSPARENT_KEY = "#010203"
PET_HEIGHT = 330

MESSAGES = (
    "今天也要开心呀～",
    "工作一会儿，记得喝水！",
    "我在桌面陪着你呢 ♡",
    "累了就休息五分钟吧。",
    "偷偷给你加个油！",
    "不许一直盯着屏幕哦～",
)


class DesktopPet:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("她的桌面小宠物")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=TRANSPARENT_KEY)
        self.root.wm_attributes("-transparentcolor", TRANSPARENT_KEY)

        source = Image.open(SPRITE_PATH).convert("RGBA")
        width = round(source.width * PET_HEIGHT / source.height)
        self.base_image = source.resize((width, PET_HEIGHT), Image.Resampling.LANCZOS)
        self.frames = {
            0: ImageTk.PhotoImage(self.base_image),
            1: ImageTk.PhotoImage(
                self.base_image.resize(
                    (width, PET_HEIGHT - 3), Image.Resampling.LANCZOS
                )
            ),
        }
        self.pet_width = width
        self.frame_index = 0
        self.drag_offset = (0, 0)
        self.bubble_job: str | None = None

        self.canvas = tk.Canvas(
            self.root,
            width=width + 12,
            height=PET_HEIGHT + 12,
            bg=TRANSPARENT_KEY,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()
        self.sprite = self.canvas.create_image(6, 6, anchor="nw", image=self.frames[0])

        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.say())
        self.canvas.bind("<Button-3>", self.show_menu)

        self.menu = tk.Menu(self.root, tearoff=False)
        self.menu.add_command(label="说句话", command=self.say)
        self.menu.add_command(label="回到右下角", command=self.move_to_corner)
        self.menu.add_separator()
        self.menu.add_command(label="退出", command=self.root.destroy)

        self.move_to_corner()
        self.root.after(700, self.breathe)
        self.root.after(random.randint(2200, 4800), self.blink)
        self.root.after(1200, lambda: self.say("我来陪你啦 ♡"))

    def move_to_corner(self) -> None:
        self.root.update_idletasks()
        x = self.root.winfo_screenwidth() - self.pet_width - 34
        y = self.root.winfo_screenheight() - PET_HEIGHT - 82
        self.root.geometry(f"+{max(0, x)}+{max(0, y)}")

    def start_drag(self, event: tk.Event) -> None:
        self.drag_offset = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def drag(self, event: tk.Event) -> None:
        x = event.x_root - self.drag_offset[0]
        y = event.y_root - self.drag_offset[1]
        self.root.geometry(f"+{x}+{y}")

    def breathe(self) -> None:
        self.frame_index = 1 - self.frame_index
        image = self.frames[self.frame_index]
        y = 7 if self.frame_index else 6
        self.canvas.itemconfigure(self.sprite, image=image)
        self.canvas.coords(self.sprite, 6, y)
        self.root.after(700, self.breathe)

    def blink(self) -> None:
        # Eye coordinates are proportional to this generated sprite.
        y = round(PET_HEIGHT * 0.315)
        left_x = round(self.pet_width * 0.39)
        right_x = round(self.pet_width * 0.61)
        half_width = max(7, round(self.pet_width * 0.08))
        self.canvas.create_line(
            left_x - half_width,
            y,
            left_x + half_width,
            y,
            fill="#5b403a",
            width=3,
            smooth=True,
            tags="blink",
        )
        self.canvas.create_line(
            right_x - half_width,
            y,
            right_x + half_width,
            y,
            fill="#5b403a",
            width=3,
            smooth=True,
            tags="blink",
        )
        self.root.after(150, lambda: self.canvas.delete("blink"))
        self.root.after(random.randint(2400, 5500), self.blink)

    def say(self, message: str | None = None) -> None:
        if self.bubble_job is not None:
            self.root.after_cancel(self.bubble_job)
        self.canvas.delete("bubble")
        text = message or random.choice(MESSAGES)
        x = self.pet_width // 2 + 6
        self.canvas.create_rectangle(
            6,
            7,
            self.pet_width + 6,
            45,
            fill="#fff7fb",
            outline="#ef9fbd",
            width=2,
            tags="bubble",
        )
        self.canvas.create_text(
            x,
            26,
            text=text,
            fill="#6d344a",
            font=("Microsoft YaHei UI", 10, "bold"),
            tags="bubble",
        )
        self.canvas.tag_raise("bubble")
        self.bubble_job = self.root.after(3200, lambda: self.canvas.delete("bubble"))

    def show_menu(self, event: tk.Event) -> None:
        self.menu.tk_popup(event.x_root, event.y_root)

    def run(self) -> None:
        self.root.mainloop()


def check_installation() -> int:
    if not SPRITE_PATH.exists():
        print(f"Missing sprite: {SPRITE_PATH}", file=sys.stderr)
        return 1
    with Image.open(SPRITE_PATH) as image:
        if image.mode != "RGBA" or image.getextrema()[3][0] == 255:
            print("Sprite does not contain a transparent alpha channel.", file=sys.stderr)
            return 1
        print(f"OK: {SPRITE_PATH.name}, {image.width}x{image.height}, RGBA")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="A tiny chibi desktop companion.")
    parser.add_argument("--check", action="store_true", help="validate assets and exit")
    args = parser.parse_args()
    if args.check:
        return check_installation()
    DesktopPet().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
