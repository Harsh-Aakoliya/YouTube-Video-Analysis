import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "YouTube Analysis",
  description: "End-to-end YouTube intelligence playground",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

