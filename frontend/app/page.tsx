"use client";

import { ArrowRight, Brain } from "lucide-react";
import { useRouter } from "next/navigation";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";

export default function Home() {
  const router = useRouter();

  return (
    <main className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center p-4">
      <div className="max-w-md w-full">
        <div className="text-center mb-8">
          <div className="inline-block p-4 rounded-full bg-primary/10 mb-4">
            <Brain className="w-12 h-12 text-primary" />
          </div>
          <h1 className="text-4xl font-bold tracking-tight mb-2">Medical AI Assistant</h1>
          <p className="text-muted-foreground">
            Advanced medical image analysis and report interpretation
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card 
            className="p-6 hover:shadow-lg transition-shadow cursor-pointer"
            onClick={() => router.push('/dashboard?role=patient')}
          >
            <h2 className="text-xl font-semibold mb-2">I'm a Patient</h2>
            <p className="text-sm text-muted-foreground mb-4">
              Get simplified explanations of medical reports and images
            </p>
            <Button className="w-full">
              Continue as Patient
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Card>

          <Card 
            className="p-6 hover:shadow-lg transition-shadow cursor-pointer"
            onClick={() => router.push('/dashboard?role=doctor')}
          >
            <h2 className="text-xl font-semibold mb-2">I'm a Doctor</h2>
            <p className="text-sm text-muted-foreground mb-4">
              Access technical analysis and professional tools
            </p>
            <Button className="w-full">
              Continue as Doctor
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Card>
        </div>
      </div>
    </main>
  );
}