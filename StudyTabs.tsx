import React from 'react';
import { 
  FileText, 
  Brain, 
  HelpCircle, 
  MessageCircle, 
  BookOpen, 
  Video, 
  Globe,
  StickyNote,
  Quote,
  Calendar,
  Network,
  BarChart3,
  Bookmark,
  Timer,
  Presentation
} from 'lucide-react';
import { motion } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import type { TabType } from '../types';

const tabs = [
  { id: 'summary' as TabType, label: 'Summary', icon: FileText, color: 'from-blue-500 to-cyan-500', bgColor: 'bg-blue-50 dark:bg-blue-900/20' },
  { id: 'flashcards' as TabType, label: 'Flashcards', icon: Brain, color: 'from-purple-500 to-pink-500', bgColor: 'bg-purple-50 dark:bg-purple-900/20' },
  { id: 'quiz' as TabType, label: 'Quiz', icon: HelpCircle, color: 'from-green-500 to-emerald-500', bgColor: 'bg-green-50 dark:bg-green-900/20' },
  { id: 'qa' as TabType, label: 'Q&A', icon: MessageCircle, color: 'from-orange-500 to-red-500', bgColor: 'bg-orange-50 dark:bg-orange-900/20' },
  { id: 'notes' as TabType, label: 'Notes', icon: StickyNote, color: 'from-yellow-500 to-amber-500', bgColor: 'bg-yellow-50 dark:bg-yellow-900/20' },
  { id: 'planner' as TabType, label: 'Study Plan', icon: Calendar, color: 'from-teal-500 to-cyan-500', bgColor: 'bg-teal-50 dark:bg-teal-900/20' },
  { id: 'timer' as TabType, label: 'Focus Timer', icon: Timer, color: 'from-rose-500 to-pink-500', bgColor: 'bg-rose-50 dark:bg-rose-900/20' },
  { id: 'research' as TabType, label: 'Research', icon: BookOpen, color: 'from-red-500 to-pink-500', bgColor: 'bg-red-50 dark:bg-red-900/20' },
  { id: 'videos' as TabType, label: 'Videos', icon: Video, color: 'from-pink-500 to-rose-500', bgColor: 'bg-pink-50 dark:bg-pink-900/20' },
  { id: 'resources' as TabType, label: 'Resources', icon: Globe, color: 'from-indigo-500 to-purple-500', bgColor: 'bg-indigo-50 dark:bg-indigo-900/20' },
  { id: 'citations' as TabType, label: 'Citations', icon: Quote, color: 'from-slate-500 to-gray-500', bgColor: 'bg-slate-50 dark:bg-slate-900/20' },
  { id: 'mindmap' as TabType, label: 'Mind Map', icon: Network, color: 'from-violet-500 to-purple-500', bgColor: 'bg-violet-50 dark:bg-violet-900/20' },
  { id: 'analytics' as TabType, label: 'Analytics', icon: BarChart3, color: 'from-emerald-500 to-green-500', bgColor: 'bg-emerald-50 dark:bg-emerald-900/20' },
  { id: 'bookmarks' as TabType, label: 'Bookmarks', icon: Bookmark, color: 'from-blue-600 to-indigo-600', bgColor: 'bg-blue-50 dark:bg-blue-900/20' },
  { id: 'presentation' as TabType, label: 'Presentations', icon: Presentation, color: 'from-indigo-500 to-purple-600', bgColor: 'bg-indigo-50 dark:bg-indigo-900/20' },
];

export function StudyTabs() {
  const { state, dispatch } = useApp();

  const handleTabChange = (tabId: TabType) => {
    dispatch({ type: 'SET_TAB', payload: tabId });
  };

  return (
    <div className="border-b border-gray-200/60 dark:border-gray-700/60 mb-16 relative">
      <nav className="flex space-x-2 overflow-x-auto scrollbar-hide pb-2">
        {tabs.map((tab, index) => {
          const Icon = tab.icon;
          const isActive = state.currentTab === tab.id;
          
          return (
            <motion.button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={`
                flex items-center space-x-3 px-10 py-5 text-sm font-bold rounded-t-3xl transition-all duration-500 whitespace-nowrap relative backdrop-blur-md
                ${isActive 
                  ? 'text-white shadow-3xl transform -translate-y-2 scale-105' 
                  : 'text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-white/70 dark:hover:bg-gray-700/70 hover:shadow-xl hover:transform hover:-translate-y-1 hover:scale-105'
                }
              `}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: index * 0.05 }}
              whileHover={{ y: -4 }}
              whileTap={{ scale: 0.95 }}
            >
              {isActive && (
                <motion.div
                  className={`absolute inset-0 bg-gradient-to-r ${tab.color} rounded-t-lg`}
                  layoutId="activeTab" 
                  transition={{ type: "spring", bounce: 0.3, duration: 0.8 }}
                />
              )}
              <div className="relative z-10 flex items-center space-x-3">
                <motion.div
                  animate={isActive ? { rotate: [0, 10, -10, 0] } : {}}
                  transition={{ duration: 2, repeat: isActive ? Infinity : 0 }}
                >
                  <Icon className="w-5 h-5" />
                </motion.div>
                <span>{tab.label}</span>
              </div>
              {isActive && (
                <motion.div 
                  className="absolute -bottom-1 left-0 right-0 h-1.5 bg-gradient-to-r from-white/90 via-white/70 to-white/90 rounded-full shadow-lg"
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ duration: 0.8, type: "spring" }}
                />
              )}
              {!isActive && (
                <motion.div
                  className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent rounded-t-3xl opacity-0 hover:opacity-100 transition-opacity duration-300"
                />
              )}
            </motion.button>
          );
        })}
      </nav>
    </div>
  );
}