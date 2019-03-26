// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
// California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
//
// This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
// redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
// Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.  You should have received a copy of the GNU General Public License along with this program;
// if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
//
// Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
// Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>
#include <jevois/Image/RawImageOps.H>
#include <jevoisbase/Components/ObjectDetection/BlobDetector.H>
#include <jevois/Image/RawImageOps.H>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <jevois/Debug/Timer.H>


// icon by Catalin Fertu in cinema at flaticon
static jevois::ParameterCategory const ParamCateg("SimpleVision Options");
JEVOIS_DECLARE_PARAMETER(leftAngle, jevois::Range<float>, "Angle range for left target",  
  jevois::Range<float>(-74, -69), ParamCateg);
JEVOIS_DECLARE_PARAMETER(rightAngle, jevois::Range<float>, "Angle range for right target", 
  jevois::Range<float>(-18, -12), ParamCateg);
JEVOIS_DECLARE_PARAMETER(targetRatio, jevois::Range<float>, "Height to width ratio for target", 
  jevois::Range<float>(1.7, 2.2), ParamCateg);
JEVOIS_DECLARE_PARAMETER(edgeThreshold, int, "Max distance for single target from edge", 45, ParamCateg);
JEVOIS_DECLARE_PARAMETER(frameWrites, int, "Write every n frames (-1 is disabled)", -1, ParamCateg);

//! JeVois sample module
/*! This module is provided as an example of how to create a new standalone module.

JeVois provides helper scripts and files to assist you in programming new modules, following two basic formats:

- if you wish to only create a single module that will execute a specific function, or a collection of such modules
where there is no shared code between the modules (i.e., each module does things that do not relate to the other
  modules), use the skeleton provided by this sample module. Here, all the code for the sample module is compiled
  into a single shared object (.so) file that is loaded by the JeVois engine when the corresponding video output
  format is selected by the host computer.

  - if you are planning to write a collection of modules with some shared algorithms among several of the modules, it
  is better to first create machine vision Components that implement the algorithms that are shared among several of
  your modules. You would then compile all your components into a first shared library (.so) file, and then compile
  each module into its own shared object (.so) file that depends on and automatically loads your shared library file
  when it is selected by the host computer. The jevoisbase library and collection of components and modules is an
  example for how to achieve that, where libjevoisbase.so contains code for Saliency, ObjectRecognition, etc
  components that are used in several modules, and each module's .so file contains only the code specific to that
  module.

  @author Patrick Hurley

  @videomapping YUYV 640 480 28.5 YUYV 640 480 28.5 Lightning SimpleVision
  @email phurley@gmail.com
  @copyright Copyright (C) 2017 by Patrick Hurley
  @mainurl http://lightningrobotics.com
  @license GPL v3
  @distribution Unrestricted
  @restrictions None */
  class SimpleVision : public jevois::Module,
                       public jevois::Parameter<leftAngle, rightAngle, targetRatio, edgeThreshold, frameWrites, jevois::module::serstyle>
  {
    public:
    int imageWidth;
    int frameCount = 0;
    int imageCount = 0;

    //! Default base class constructor ok
    // using jevois::Module::Module;

    SimpleVision(std::string name) : jevois::Module::Module(name) {
      itsDetector = addSubComponent<BlobDetector>("detector");
      frameCount = 0;
      imageCount = 0;
    }

    //! Virtual destructor for safe inheritance
    virtual ~SimpleVision() { }

    //! Processing function, no USB video output
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image. Any resolution and format ok:
      jevois::RawImage inimg = inframe.get();
      imageWidth = inimg.width;

      // Convert input image to BGR24, then to HSV:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
      cv::Mat imghsv; cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Detect blobs and get their contours:
      auto contours = itsDetector->detect(imghsv);
      auto rects = filterContours(contours);

      if (frameWrites::get() != -1) {
        if (++frameCount % frameWrites::get() == 0) {
          cv::imwrite( "/jevois/data/simpleFrame" + std::to_string(++imageCount % 1000) + ".jpg", imgbgr);
        }
      }

      sendObjects((size_t) contours.size(), rects);
    }

    bool epsilonEqual(float v1, float v2, float epsilon) {
      return std::abs(v1 - v2) < epsilon;
    }

    bool looksLikeLeft(const cv::RotatedRect& r) {
      return leftAngle::get().contains(r.angle);
    }

    bool looksLikeRight(const cv::RotatedRect& r) {
      return rightAngle::get().contains(r.angle);
    }

    std::vector<cv::RotatedRect> filterContours(const std::vector<std::vector<cv::Point> >& contours) {
      std::vector<cv::RotatedRect> rects;
      int i = 0;
      for (const auto& c : contours) {
        auto r = cv::minAreaRect(c);
        auto ratio = getRatio(r);

        if (serstyle::get() == jevois::module::SerStyle::Detail) {
          sendSerial("OB" + std::to_string(i) + " " + 
            std::to_string(r.center.x) + " " + 
            std::to_string(r.center.y) + " " + 
            std::to_string(r.size.width) + " " + 
            std::to_string(r.size.height) + " " +
            std::to_string(ratio) + " " + 
            std::to_string(r.angle) + " " + 
            std::to_string(r.size.width * r.size.height));
        }
        
        if (targetRatio::get().contains(ratio)) {
          if (looksLikeLeft(r) || looksLikeRight(r)) {
            rects.push_back(cv::minAreaRect(c));
          }
        }
      }

      if (rects.size() == 1) {
        auto r = rects.front();
        // check if we look like left that we are near the right edge of the image
        if (looksLikeLeft(r) && r.center.x > edgeThreshold::get()) {
          return std::vector<cv::RotatedRect>();
        }

        // check if we look like right that we are near the left edge of the image
        if (looksLikeRight(r) && r.center.x < (imageWidth - edgeThreshold::get())) {
          return std::vector<cv::RotatedRect>();
        }
      } else if (rects.size() == 2) {
        // verify the left one is on the left and the right one is on the right
        auto r1 = rects.front();
        auto r2 = rects.back();
        if (r1.center.x > r2.center.x) {
          auto tmp = r1;
          r1 = r2;
          r2 = tmp;
        }

        if (!(looksLikeLeft(r1) && looksLikeRight(r2))) {
          return std::vector<cv::RotatedRect>();
        }
      } else if (rects.size() > 2) {
        int middle = imageWidth / 2;
        std::sort(rects.begin(), rects.end(), [middle](auto r1, auto r2) { 
          return std::abs(middle - r1.center.x) > 
          std::abs(middle - r2.center.x); 
        });

        auto r1 = rects[0];
        auto r2 = rects[1];
        if (r1.center.x > r2.center.x) {
          auto tmp = r1;
          r1 = r2;
          r2 = tmp;
        }

        if (!(looksLikeLeft(r1) && looksLikeRight(r2))) {
          return std::vector<cv::RotatedRect>();
        }
        rects.resize(2);
      }

      return rects;
    }

    float getRatio(const cv::Size2f& s) {
      return std::max(s.width, s.height) / std::min(s.width, s.height);
    }

    float getRatio(const cv::RotatedRect& s) {
      return getRatio(s.size);
    }

    void sendRectangle(size_t count, const cv::Rect& r) {
      sendSerial("SV" + std::to_string(count) + " " + 
        std::to_string(r.x + 0.5F * r.width) + " " + 
        std::to_string(r.y + 0.5F + r.height) + " " +
        std::to_string(r.width) + " " + 
        std::to_string(r.height));
    }

    void sendObjects(size_t count, const std::vector<cv::RotatedRect>& rects) {
      switch (rects.size()) {
        case 0:
          sendSerial("SV0 0 0 0 0");
          break;

        case 1:
          sendRectangle(1, rects.front().boundingRect());
          {
            cv::RotatedRect rr = rects.front();
            sendSerial("ANGLE1 " + std::to_string(rr.angle));
            sendSerial("HEIGHT1 " + std::to_string(rr.size.height));
            sendSerial("WIDTH1 " + std::to_string(rr.size.width));
            sendSerial("RATIO1 " + std::to_string(getRatio(rr.size)));
          }
          break;

        default:
          {
            cv::RotatedRect r1 = rects[0];
            sendSerial("ANGLE1 " + std::to_string(r1.angle));
            sendSerial("HEIGHT1 " + std::to_string(r1.size.height));
            sendSerial("WIDTH1 " + std::to_string(r1.size.width));
            sendSerial("RATIO1 " + std::to_string(getRatio(r1.size)));
            cv::RotatedRect r2 = rects[1];
            sendSerial("ANGLE2 " + std::to_string(r2.angle));
            sendSerial("HEIGHT2 " + std::to_string(r2.size.height));
            sendSerial("WIDTH2 " + std::to_string(r2.size.width));
            sendSerial("RATIO2 " + std::to_string(getRatio(r2.size)));

            auto r = r1.boundingRect() | r2.boundingRect();
            sendRectangle(count, r);
          }
          break;
      }
    }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing");

      // Wait for next available camera image. Any resolution ok, but require YUYV since we assume it for drawings:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      imageWidth = w;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      timer.start();

      // While we process it, start a thread to wait for output frame and paste the input image into it:
      jevois::RawImage outimg; // main thread should not use outimg until paste thread is complete
      auto paste_fut = std::async(std::launch::async, [&]() {
        outimg = outframe.get();
        outimg.require("output", w, h + 14, inimg.fmt);
        jevois::rawimage::paste(inimg, outimg, 0, 0);
        jevois::rawimage::writeText(outimg, "Lightning Deepspace Tracker", 3, 3, jevois::yuyv::White);
        jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, 0x8000);
      });

      // Convert input image to BGR24, then to HSV:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
      cv::Mat imghsv; cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);

      // Detect blobs and get their contours:
      auto contours = itsDetector->detect(imghsv);
      auto rects = filterContours(contours);

      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();

      // Draw all detected contours in a thread:
      std::future<void> draw_fut = std::async(std::launch::async, [&]() {
        // We reinterpret the top portion of our YUYV output image as an opencv 8UC2 image:
        cv::Mat outuc2 = jevois::rawimage::cvImage(outimg); // pixel data shared
        cv::drawContours(outuc2, contours, -1, jevois::yuyv::LightPurple, 2, 8);
      });

      sendObjects(contours.size(), rects);

      for (auto const & rr : rects)
      {
        cv::Point2f points[4];
        rr.points(points);
        for (int i = 0; i < 4; ++i) {
          auto p1 = points[i];
          auto p2 = points[(i + 1) % 4];
          jevois::rawimage::drawLine(outimg, p1.x, p1.y, p2.x, p2.y, 2, jevois::yuyv::DarkPink);
        }
      }

      // Show number of detected objects:
      jevois::rawimage::writeText(outimg, "Detected " + std::to_string(contours.size()) + "/" + 
        std::to_string(rects.size()) + " objects.", 3, h + 2, jevois::yuyv::White);

        // Show processing fps:
        std::string const & fpscpu = timer.stop();
        jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

        // Wait until all contours are drawn, if they had been requested:
        draw_fut.get();

        if (frameWrites::get() != -1) {
            std::cout << "Check " << frameCount << std::endl;
          if (++frameCount % frameWrites::get() == 0) {
            std::cout << "Writing " << imageCount << std::endl;
            cv::imwrite( std::string("/jevois/data/simpleFrame") + std::to_string(++imageCount % 1000) + ".jpg", 
              imgbgr);
          }
        }

        // Send the output image with our processing results to the host over USB:
        outframe.send();
    }

    private:
    std::shared_ptr<BlobDetector> itsDetector;
  };

  // Allow the module to be loaded as a shared object (.so) file:
  JEVOIS_REGISTER_MODULE(SimpleVision);
