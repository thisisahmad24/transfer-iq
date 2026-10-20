import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TransferIQ — Football, valued intelligently",
  description:
    "Explore data-informed Premier League transfer fee estimates powered by machine learning.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
