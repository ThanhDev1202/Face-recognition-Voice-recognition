import customtkinter as ctk
from tkinter import messagebox

from FaceVerifyWindow import FaceVerifyWindow
from VoiceVerifyWindow import VoiceVerifyWindow


class LoginWindow(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Đăng nhập MFA")
        self.geometry("500x400")

        self.face_verified = False
        self.voice_verified = False

        # =====================
        # Username
        # =====================

        ctk.CTkLabel(
            self,
            text="Tên đăng nhập",
            font=("Arial",18,"bold")
        ).pack(pady=(30,10))

        self.entry_user = ctk.CTkEntry(
            self,
            width=250
        )
        self.entry_user.pack()

        # =====================
        # Face Button
        # =====================

        self.btn_face = ctk.CTkButton(
            self,
            text="Xác thực khuôn mặt",
            command=self.verify_face
        )

        self.btn_face.pack(pady=20)

        # =====================
        # Voice Button
        # =====================

        self.btn_voice = ctk.CTkButton(
            self,
            text="Xác thực giọng nói",
            command=self.verify_voice,
            state="disabled"
        )

        self.btn_voice.pack()

        # =====================
        # Login
        # =====================

        self.btn_login = ctk.CTkButton(
            self,
            text="Đăng nhập",
            command=self.login,
            state="disabled"
        )

        self.btn_login.pack(pady=30)

    #####################################################

    def verify_face(self):

        username = self.entry_user.get()

        if username == "":
            messagebox.showwarning(
                "Thông báo",
                "Vui lòng nhập tên đăng nhập."
            )
            return

        # Mở cửa sổ xác thực khuôn mặt
        win = FaceVerifyWindow(self, username)

        self.wait_window(win)

        if getattr(win, "result", False):

            self.face_verified = True

            self.btn_voice.configure(state="normal")

            messagebox.showinfo(
                "Thông báo",
                "Xác thực khuôn mặt thành công."
            )

    #####################################################

    def verify_voice(self):

        username = self.entry_user.get()

        win = VoiceVerifyWindow(self, username)

        self.wait_window(win)

        if getattr(win, "result", False):

            self.voice_verified = True

            self.btn_login.configure(state="normal")

            messagebox.showinfo(
                "Thông báo",
                "Xác thực giọng nói thành công."
            )

    #####################################################

    def login(self):

        if not self.face_verified:

            messagebox.showerror(
                "Lỗi",
                "Chưa xác thực khuôn mặt."
            )
            return

        if not self.voice_verified:

            messagebox.showerror(
                "Lỗi",
                "Chưa xác thực giọng nói."
            )
            return

        messagebox.showinfo(
            "Thành công",
            "Đăng nhập thành công!"
        )

        self.destroy()


if __name__ == "__main__":

    app = LoginWindow()

    app.mainloop()